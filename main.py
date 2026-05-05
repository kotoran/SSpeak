#!/usr/bin/env python3
"""
SSpeak: Kokoro TTS into a real PipeWire/Pulse virtual microphone.

Design goals:
    - Rock-solid on Linux Mint and Arch with PipeWire-Pulse.
    - No GUI.
    - Offline-first: local model + local voice .pt files by default.
    - sounddevice is only a fallback speaker backend; it does not expose a mic.
    - The exposed microphone route is created with PipeWire ports and pw-link.

Default route:
    Kokoro audio -> paplay --device=SSpeakSpeaker
                 -> SSpeakSpeaker.monitor
                 -> pw-link
                 -> SSpeakMic
                 -> Discord / OBS / Zoom / browser / game

Common commands:
    python main.py
    python main.py --no-repl "Speak once and exit."
    python main.py --no-default-source --no-move-recording-streams
    python main.py --cleanup-on-exit
    python main.py --setup-mic
    python main.py --test-route
    python main.py --monitor "Hello."
    python main.py --status
    python main.py --teardown-mic
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import wave
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import soundfile as sf


APP_NAME = "SSpeak"
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_ROUTE_SAMPLE_RATE = 48_000
DEFAULT_SPEAKER_SINK = "SSpeakSpeaker"
DEFAULT_MIC_SOURCE = "SSpeakMic"
DEFAULT_MONITOR_TOKEN = "SSpeakLocalMonitor"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "model" / "config.json"
DEFAULT_MODEL_FILE = PROJECT_ROOT / "model" / "kokoro-v1_0.pth"
DEFAULT_VOICE_DIR = PROJECT_ROOT / "voices"

VOICE_LANG_PREFIXES = {
    "af_": "a",
    "am_": "a",
    "bf_": "b",
    "bm_": "b",
    "jf_": "j",
    "jm_": "j",
    "zf_": "z",
    "zm_": "z",
    "ef_": "e",
    "em_": "e",
    "ff_": "f",
    "fm_": "f",
    "hf_": "h",
    "hm_": "h",
    "if_": "i",
    "im_": "i",
    "pf_": "p",
    "pm_": "p",
}


class SSpeakError(RuntimeError):
    """Raised for expected user-facing errors."""


@dataclass(frozen=True)
class CommandResult:
    """A completed subprocess command."""

    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Thin subprocess wrapper with readable error messages."""

    def __init__(self, *, verbose: bool = False, dry_run: bool = False) -> None:
        self._verbose = verbose
        self._dry_run = dry_run

    @staticmethod
    def which(program: str) -> Optional[str]:
        """Return executable path or None."""
        return shutil.which(program)

    def run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        capture: bool = True,
        input_text: Optional[str] = None,
    ) -> CommandResult:
        """Run a command."""
        printable = " ".join(str(arg) for arg in args)

        if self._verbose or self._dry_run:
            print(f"+ {printable}", file=sys.stderr)

        if self._dry_run:
            return CommandResult(args=args, returncode=0, stdout="", stderr="")

        completed = subprocess.run(
            [str(arg) for arg in args],
            check=False,
            text=True,
            input=input_text,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if check and completed.returncode != 0:
            message = (
                f"Command failed with exit code {completed.returncode}:\n"
                f"  {printable}"
            )

            if stdout.strip():
                message += f"\n\nstdout:\n{stdout}"

            if stderr.strip():
                message += f"\n\nstderr:\n{stderr}"

            raise SSpeakError(message)

        return CommandResult(
            args=args,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )


@dataclass(frozen=True)
class PulseModule:
    """One row from `pactl list modules short`."""

    index: int
    name: str
    arguments: str

    @staticmethod
    def parse(line: str) -> Optional["PulseModule"]:
        """Parse a module line."""
        parts = line.split("\t")

        if len(parts) < 2:
            parts = re.split(r"\s+", line, maxsplit=2)

        if len(parts) < 2:
            return None

        try:
            index = int(parts[0])
        except ValueError:
            return None

        name = parts[1]
        arguments = parts[2] if len(parts) > 2 else ""
        return PulseModule(index=index, name=name, arguments=arguments)

    def contains(self, token: str) -> bool:
        """Return True if token appears in module name or arguments."""
        return token in f"{self.name} {self.arguments}"


class PulsePipeWireRoute:
    """Create SSpeakSpeaker -> SSpeakMic with PipeWire pw-link."""

    def __init__(
        self,
        runner: CommandRunner,
        *,
        speaker_sink: str,
        mic_source: str,
        monitor_token: str = DEFAULT_MONITOR_TOKEN,
        route_sample_rate: int = DEFAULT_ROUTE_SAMPLE_RATE,
    ) -> None:
        self._runner = runner
        self._speaker_sink = speaker_sink
        self._mic_source = mic_source
        self._monitor_token = monitor_token
        self._route_sample_rate = route_sample_rate

    @property
    def speaker_sink(self) -> str:
        """Return speaker sink name."""
        return self._speaker_sink

    @property
    def mic_source(self) -> str:
        """Return mic source name."""
        return self._mic_source

    def require_available(self) -> None:
        """Require Pulse/PipeWire tooling."""
        missing = [
            program
            for program in ("pactl", "paplay", "parecord", "pw-link")
            if self._runner.which(program) is None
        ]

        if missing:
            raise SSpeakError(
                "Missing required audio tools: "
                + ", ".join(missing)
                + "\nMint: sudo apt install pulseaudio-utils pipewire-bin"
                + "\nArch: sudo pacman -S --needed libpulse pipewire"
            )

        result = self._runner.run(["pactl", "info"], check=False, capture=True)
        if result.returncode != 0:
            raise SSpeakError(
                "pactl cannot reach the PulseAudio-compatible server.\n"
                "Make sure PipeWire-Pulse is running:\n"
                "  systemctl --user enable --now pipewire pipewire-pulse wireplumber"
            )

    def is_available(self) -> bool:
        """Return True if required tools and server are available."""
        try:
            self.require_available()
        except SSpeakError:
            return False

        return True

    def list_modules(self) -> list[PulseModule]:
        """List loaded Pulse modules."""
        result = self._runner.run(
            ["pactl", "list", "modules", "short"],
            check=True,
            capture=True,
        )
        modules: list[PulseModule] = []

        for line in result.stdout.splitlines():
            module = PulseModule.parse(line)
            if module is not None:
                modules.append(module)

        return modules

    def module_loaded_with_token(self, token: str) -> bool:
        """Return True if any module contains token."""
        return any(module.contains(token) for module in self.list_modules())

    def sink_exists(self) -> bool:
        """Return True if SSpeakSpeaker exists."""
        result = self._runner.run(
            ["pactl", "list", "short", "sinks"],
            check=False,
            capture=True,
        )
        return self._speaker_sink in result.stdout

    def source_exists(self) -> bool:
        """Return True if SSpeakMic exists."""
        result = self._runner.run(
            ["pactl", "list", "short", "sources"],
            check=False,
            capture=True,
        )
        return self._mic_source in result.stdout

    def setup(self, *, recreate: bool = False) -> None:
        """Create and link the virtual route."""
        self.require_available()

        # The old module-remap-source route can expose a source that remains silent
        # on some PipeWire-Pulse systems. Remove it and rebuild with pw-link.
        old_remap_loaded = self.module_loaded_with_token(f"source_name={self._mic_source}")

        if recreate or old_remap_loaded:
            self.teardown()

        self._load_speaker_sink_if_needed()
        self._load_virtual_source_if_needed()
        self._link_monitor_to_virtual_source()
        self._unmute_and_set_levels()

    def teardown(self) -> None:
        """Unload SSpeak-managed modules."""
        self.require_available()

        tokens = (
            f"sink_name={self._speaker_sink}",
            f"sink_name={self._mic_source}",
            f"source_name={self._mic_source}",
            self._monitor_token,
            f"source={self._speaker_sink}.monitor",
        )

        for module in self.list_modules():
            if any(module.contains(token) for token in tokens):
                self._runner.run(
                    ["pactl", "unload-module", str(module.index)],
                    check=False,
                    capture=True,
                )

    def monitor_on(self) -> None:
        """Loop SSpeakSpeaker monitor to local default speakers."""
        self.setup()

        if self.module_loaded_with_token(self._monitor_token):
            return

        self._runner.run(
            [
                "pactl",
                "load-module",
                "module-loopback",
                f"source={self._speaker_sink}.monitor",
                "sink=@DEFAULT_SINK@",
                "latency_msec=30",
                f"sink_input_properties=application.name={self._monitor_token}",
            ],
            check=True,
            capture=True,
        )

    def monitor_off(self) -> None:
        """Disable local monitoring loopback."""
        self.require_available()

        for module in self.list_modules():
            if module.contains(self._monitor_token):
                self._runner.run(
                    ["pactl", "unload-module", str(module.index)],
                    check=False,
                    capture=True,
                )

    def print_status(self) -> None:
        """Print route status."""
        self.require_available()

        print(f"Virtual speaker sink: {self._speaker_sink}")
        print(f"Virtual mic source:   {self._mic_source}")
        print(f"Sink exists:          {self.sink_exists()}")
        print(f"Source exists:        {self.source_exists()}")

        print("\nRelevant modules:")
        found = False
        for module in self.list_modules():
            if (
                module.contains(self._speaker_sink)
                or module.contains(self._mic_source)
                or module.contains(self._monitor_token)
            ):
                found = True
                print(f"  {module.index}: {module.name} {module.arguments}")

        if not found:
            print("  none")

        print("\nPipeWire links containing SSpeak:")
        links = self._runner.run(["pw-link", "-l"], check=False, capture=True)
        printed = False
        for line in links.stdout.splitlines():
            if self._speaker_sink in line or self._mic_source in line:
                printed = True
                print(line)

        if not printed:
            print("  none")

        print("\nSinks:")
        self._runner.run(["pactl", "list", "short", "sinks"], check=False, capture=False)

        print("\nSources:")
        self._runner.run(
            ["pactl", "list", "short", "sources"],
            check=False,
            capture=False,
        )

    def test_route(self, *, output_path: Path) -> bool:
        """Generate a tone, route it through SSpeakMic, record, and check signal."""
        self.setup(recreate=True)

        with tempfile.TemporaryDirectory(prefix="sspeak-route-test-") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            tone_path = temp_dir / "tone.wav"
            record_path = output_path

            self._write_test_tone(tone_path)

            record_process = subprocess.Popen(
                [
                    "parecord",
                    f"--device={self._mic_source}",
                    "--file-format=wav",
                    str(record_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )

            try:
                time.sleep(0.75)
                self._runner.run(
                    ["paplay", f"--device={self._speaker_sink}", str(tone_path)],
                    check=True,
                    capture=True,
                )
                time.sleep(0.75)
            finally:
                self._terminate_process(record_process)

            peak, rms = self._measure_wav(record_path)
            print(f"Recorded: {record_path}")
            print(f"Peak:     {peak}")
            print(f"RMS:      {rms:.2f}")

            if peak <= 0:
                print("FAILED: route recording is digital silence.", file=sys.stderr)
                return False

            print("OK: SSpeakMic captured audio.")
            return True

    def _load_speaker_sink_if_needed(self) -> None:
        if self.sink_exists():
            return

        self._runner.run(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={self._speaker_sink}",
                f"sink_properties=device.description={self._speaker_sink}",
                f"rate={self._route_sample_rate}",
                "channels=2",
                "channel_map=front-left,front-right",
            ],
            check=True,
            capture=True,
        )

    def _load_virtual_source_if_needed(self) -> None:
        if self.source_exists():
            return

        self._runner.run(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                "media.class=Audio/Source/Virtual",
                f"sink_name={self._mic_source}",
                f"sink_properties=device.description={self._mic_source}",
                f"rate={self._route_sample_rate}",
                "channels=2",
                "channel_map=front-left,front-right",
            ],
            check=True,
            capture=True,
        )

    def _link_monitor_to_virtual_source(self) -> None:
        outputs = self._list_pw_ports(["pw-link", "-o"])
        inputs = self._list_pw_ports(["pw-link", "-i"])

        out_l = self._pick_port(outputs, self._speaker_sink, ("monitor_FL", "monitor_L", "FL", "left"))
        out_r = self._pick_port(outputs, self._speaker_sink, ("monitor_FR", "monitor_R", "FR", "right"))
        in_l = self._pick_port(inputs, self._mic_source, ("input_FL", "input_L", "FL", "left"))
        in_r = self._pick_port(inputs, self._mic_source, ("input_FR", "input_R", "FR", "right"))

        self._runner.run(["pw-link", out_l, in_l], check=False, capture=True)
        self._runner.run(["pw-link", out_r, in_r], check=False, capture=True)

    def _unmute_and_set_levels(self) -> None:
        self._runner.run(
            ["pactl", "set-sink-mute", self._speaker_sink, "0"],
            check=False,
            capture=True,
        )
        self._runner.run(
            ["pactl", "set-sink-volume", self._speaker_sink, "100%"],
            check=False,
            capture=True,
        )
        self._runner.run(
            ["pactl", "set-source-mute", self._mic_source, "0"],
            check=False,
            capture=True,
        )
        self._runner.run(
            ["pactl", "set-source-volume", self._mic_source, "100%"],
            check=False,
            capture=True,
        )

    def set_default_source(self) -> None:
        """Make SSpeakMic the default input for newly opened apps."""
        self.require_available()

        if not self.source_exists():
            raise SSpeakError(
                f"Cannot set default source because {self._mic_source!r} does not exist."
            )

        self._runner.run(
            ["pactl", "set-default-source", self._mic_source],
            check=False,
            capture=True,
        )
        self._runner.run(
            ["pactl", "set-source-mute", self._mic_source, "0"],
            check=False,
            capture=True,
        )
        self._runner.run(
            ["pactl", "set-source-volume", self._mic_source, "125%"],
            check=False,
            capture=True,
        )

    def move_recording_streams_to_mic(self) -> int:
        """Move currently active recording streams to SSpeakMic.

        This makes already-open apps such as Discord switch to the virtual mic
        without requiring pavucontrol in many cases. Apps can still override this
        internally, but Pulse/PipeWire streams that allow moving will be moved.
        """
        self.require_available()

        result = self._runner.run(
            ["pactl", "list", "short", "source-outputs"],
            check=False,
            capture=True,
        )

        moved = 0

        for line in result.stdout.splitlines():
            parts = line.split()

            if not parts:
                continue

            source_output_id = parts[0]

            move_result = self._runner.run(
                ["pactl", "move-source-output", source_output_id, self._mic_source],
                check=False,
                capture=True,
            )

            if move_result.returncode == 0:
                moved += 1

        return moved

    def get_default_source(self) -> Optional[str]:
        """Return the current default input source name, if available."""
        self.require_available()
        result = self._runner.run(["pactl", "info"], check=False, capture=True)

        for line in result.stdout.splitlines():
            if line.startswith("Default Source:"):
                return line.split(":", 1)[1].strip()

        return None

    def source_name_exists(self, source_name: str) -> bool:
        """Return True if a source exists by exact Pulse/PipeWire name."""
        result = self._runner.run(
            ["pactl", "list", "short", "sources"],
            check=False,
            capture=True,
        )

        for line in result.stdout.splitlines():
            parts = line.split()

            if len(parts) >= 2 and parts[1] == source_name:
                return True

        return False

    def find_fallback_source(self) -> Optional[str]:
        """Find a non-SSpeak, non-monitor source to restore after cleanup."""
        result = self._runner.run(
            ["pactl", "list", "short", "sources"],
            check=False,
            capture=True,
        )

        for line in result.stdout.splitlines():
            parts = line.split()

            if len(parts) < 2:
                continue

            source_name = parts[1]
            lowered = source_name.lower()

            if self._speaker_sink.lower() in lowered:
                continue

            if self._mic_source.lower() in lowered:
                continue

            if ".monitor" in lowered:
                continue

            return source_name

        return None

    def restore_default_source(self, previous_source: Optional[str]) -> Optional[str]:
        """Restore default source to the previous mic or a sane fallback.

        If the previous source was SSpeakMic from an earlier run, restoring it
        before teardown would leave the desktop with a deleted default source.
        In that case, this chooses the first real non-monitor input source.
        """
        self.require_available()

        target = None

        if (
            previous_source
            and previous_source != self._mic_source
            and self.source_name_exists(previous_source)
        ):
            target = previous_source
        else:
            target = self.find_fallback_source()

        if not target:
            return None

        self._runner.run(
            ["pactl", "set-default-source", target],
            check=False,
            capture=True,
        )
        return target

    def _list_pw_ports(self, args: Sequence[str]) -> list[str]:
        result = self._runner.run(args, check=True, capture=True)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    @staticmethod
    def _pick_port(
        ports: Sequence[str],
        owner: str,
        preferred_tokens: Sequence[str],
    ) -> str:
        owner_matches = [port for port in ports if owner.lower() in port.lower()]

        for token in preferred_tokens:
            token_matches = [
                port for port in owner_matches if token.lower() in port.lower()
            ]
            if token_matches:
                return token_matches[0]

        raise SSpeakError(
            f"Could not find PipeWire port for {owner}.\n"
            f"Tokens tried: {', '.join(preferred_tokens)}\n"
            "Matching ports:\n"
            + "\n".join(owner_matches)
        )

    @staticmethod
    def _write_test_tone(path: Path) -> None:
        sample_rate = DEFAULT_ROUTE_SAMPLE_RATE
        seconds = 3
        frequency = 440.0
        amplitude = 0.35

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for index in range(sample_rate * seconds):
                value = int(
                    amplitude
                    * 32767
                    * math.sin(2 * math.pi * frequency * index / sample_rate)
                )
                wav_file.writeframes(struct.pack("<hh", value, value))

    @staticmethod
    def _measure_wav(path: Path) -> tuple[int, float]:
        with wave.open(str(path), "rb") as wav_file:
            data = wav_file.readframes(wav_file.getnframes())

        samples = np.frombuffer(data, dtype="<i2")

        if samples.size == 0:
            return 0, 0.0

        peak = int(np.abs(samples).max())
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        return peak, rms

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=3)


class AudioSink(ABC):
    """Abstract destination for generated mono float32 audio."""

    @abstractmethod
    def open(self) -> None:
        """Prepare the sink."""

    @abstractmethod
    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play or write one utterance."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class PulseVirtualMicSink(AudioSink):
    """Play generated audio into SSpeakSpeaker with paplay."""

    def __init__(
        self,
        runner: CommandRunner,
        route: PulsePipeWireRoute,
        *,
        setup_route: bool,
    ) -> None:
        self._runner = runner
        self._route = route
        self._setup_route = setup_route

    def open(self) -> None:
        """Ensure route exists unless automatic setup was disabled."""
        if self._setup_route:
            self._route.setup()
            return

        self._route.require_available()

        if not self._route.sink_exists():
            raise SSpeakError(
                f"Virtual sink {self._route.speaker_sink!r} does not exist. "
                "Run with --setup-mic first, or remove --no-setup-mic."
            )

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Write temporary WAV and paplay it into the virtual sink."""
        temp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)

            sf.write(temp_path, audio, sample_rate)
            self._runner.run(
                ["paplay", f"--device={self._route.speaker_sink}", str(temp_path)],
                check=True,
                capture=True,
            )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def close(self) -> None:
        """No per-session resources to close."""


class SoundDeviceSink(AudioSink):
    """Cross-platform speaker playback fallback. It does not expose a mic."""

    def __init__(self, device: Optional[str]) -> None:
        self._device_spec = device
        self._device: int | str | None = None
        self._sounddevice = None

    def open(self) -> None:
        """Import and configure sounddevice."""
        try:
            import sounddevice as sd  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise SSpeakError(
                "sounddevice is not installed. Install it or use --backend pulse."
            ) from exc

        self._sounddevice = sd
        self._device = self._resolve_output_device(self._device_spec)

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio to speakers only."""
        if self._sounddevice is None:
            raise SSpeakError("SoundDeviceSink was not opened.")

        self._sounddevice.play(
            audio,
            samplerate=sample_rate,
            device=self._device,
            blocking=True,
        )

    def close(self) -> None:
        """Stop playback."""
        if self._sounddevice is not None:
            self._sounddevice.stop()

    def _resolve_output_device(self, spec: Optional[str]) -> int | str | None:
        if self._sounddevice is None or spec is None:
            return None

        if spec.isdigit():
            return int(spec)

        devices = self._sounddevice.query_devices()
        matches: list[tuple[int, str]] = []

        for index, device in enumerate(devices):
            name = str(device.get("name", ""))
            output_channels = int(device.get("max_output_channels", 0))
            if output_channels > 0 and spec.lower() in name.lower():
                matches.append((index, name))

        if not matches:
            raise SSpeakError(f"No output device contains {spec!r}. Use --list-devices.")

        if len(matches) > 1:
            choices = "\n".join(f"  {index}: {name}" for index, name in matches)
            raise SSpeakError(f"Multiple output devices match {spec!r}:\n{choices}")

        index, name = matches[0]
        print(f"Using output device [{index}] {name}", file=sys.stderr)
        return index


class FileSink(AudioSink):
    """Write generated audio to WAV files."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._counter = 0

    def open(self) -> None:
        """Ensure destination exists."""
        if self._output_path.suffix.lower() == ".wav":
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._output_path.mkdir(parents=True, exist_ok=True)

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Write one utterance."""
        if self._output_path.suffix.lower() == ".wav" and self._counter == 0:
            path = self._output_path
        elif self._output_path.suffix.lower() == ".wav":
            path = self._output_path.with_name(
                f"{self._output_path.stem}_{self._counter:04d}"
                f"{self._output_path.suffix}"
            )
        else:
            path = self._output_path / f"sspeak_{self._counter:04d}.wav"

        sf.write(path, audio, sample_rate)
        print(f"Wrote {path}", file=sys.stderr)
        self._counter += 1

    def close(self) -> None:
        """No resources to close."""


class KokoroSynthesizer:
    """Load local Kokoro model and synthesize text."""

    def __init__(
        self,
        *,
        lang_code: str,
        voice: str,
        speed: float,
        model_config: Path,
        model_file: Path,
        repo_id: str,
        voice_dir: Path,
        offline: bool,
        show_warnings: bool,
    ) -> None:
        self._lang_code = lang_code
        self._voice = voice
        self._speed = speed
        self._model_config = model_config
        self._model_file = model_file
        self._repo_id = repo_id
        self._voice_dir = voice_dir
        self._offline = offline
        self._show_warnings = show_warnings
        self._voice_tensor = None
        self._pipeline = None

    def load(self) -> None:
        """Load Kokoro model and voice tensor."""
        if not self._model_config.exists():
            raise SSpeakError(f"Missing model config: {self._model_config}")

        if not self._model_file.exists():
            raise SSpeakError(f"Missing model file: {self._model_file}")

        if self._offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if not self._show_warnings:
            self._suppress_known_torch_warnings()

        from kokoro import KModel, KPipeline  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        voice_file = self._resolve_voice_file()

        print("Loading Kokoro model from local files...", file=sys.stderr)
        print(f"Model: {self._model_file}", file=sys.stderr)
        print(f"Voice: {voice_file}", file=sys.stderr)

        model = KModel(
            repo_id=self._repo_id,
            config=str(self._model_config),
            model=str(self._model_file),
        )
        self._voice_tensor = self._load_voice_tensor(torch, voice_file)
        self._pipeline = KPipeline(
            lang_code=self._resolved_lang_code(),
            model=model,
            repo_id=self._repo_id,
        )

    def synthesize(self, text: str) -> np.ndarray:
        """Return synthesized utterance as mono float32 audio."""
        if self._pipeline is None:
            self.load()

        if self._pipeline is None:
            raise SSpeakError("Kokoro pipeline failed to load.")

        if self._voice_tensor is None:
            raise SSpeakError("Local Kokoro voice tensor was not loaded.")

        generator = self._pipeline(
            text,
            voice=self._voice_tensor,
            speed=self._speed,
            split_pattern=r"\n+",
        )
        chunks: list[np.ndarray] = []

        for _graphemes, _phonemes, audio in generator:
            chunks.append(self._as_numpy_f32(audio))

        if not chunks:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _resolved_lang_code(self) -> str:
        if self._lang_code != "auto":
            return self._lang_code

        for prefix, lang_code in VOICE_LANG_PREFIXES.items():
            if self._voice.startswith(prefix):
                return lang_code

        return "a"

    def _resolve_voice_file(self) -> Path:
        voice_path = Path(self._voice).expanduser()

        if voice_path.exists():
            return voice_path.resolve()

        if voice_path.suffix == ".pt":
            candidate = self._voice_dir / voice_path.name
        else:
            candidate = self._voice_dir / f"{self._voice}.pt"

        if not candidate.exists():
            raise SSpeakError(
                f"Missing local voice file: {candidate}\n"
                "Offline mode is enabled, so SSpeak will not download voices.\n"
                "Use --voice-dir /path/to/voices, use --voice /path/to/voice.pt, "
                "or pass --online if network fallback is intentional."
            )

        return candidate.resolve()

    @staticmethod
    def _load_voice_tensor(torch_module: object, voice_file: Path) -> object:
        try:
            return torch_module.load(
                str(voice_file),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch_module.load(str(voice_file), map_location="cpu")

    @staticmethod
    def _as_numpy_f32(audio: object) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        return np.asarray(audio, dtype=np.float32).reshape(-1)

    @staticmethod
    def _suppress_known_torch_warnings() -> None:
        warnings.filterwarnings(
            "ignore",
            message="dropout option adds dropout after all but last recurrent layer.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
            category=FutureWarning,
        )


class SSpeakApplication:
    """Command-line application."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._runner = CommandRunner(verbose=args.verbose, dry_run=args.dry_run)
        self._route = PulsePipeWireRoute(
            self._runner,
            speaker_sink=args.speaker_sink,
            mic_source=args.mic_source,
        )

    def run(self) -> int:
        """Run the requested command."""
        try:
            return self._run_or_raise()
        except SSpeakError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130

    def _run_or_raise(self) -> int:
        if self._args.install_hints:
            self._print_install_hints()
            return 0

        if self._args.list_devices:
            self._list_sounddevice_devices()
            return 0

        if self._args.setup_mic:
            self._route.setup(recreate=self._args.recreate_route)
            print(f"Created/verified microphone source: {self._args.mic_source}")
            return 0

        if self._args.teardown_mic:
            self._route.teardown()
            print("Removed SSpeak virtual microphone route.")
            return 0

        if self._args.status:
            self._route.print_status()
            return 0

        if self._args.test_route:
            output_path = Path(self._args.test_output).expanduser()
            ok = self._route.test_route(output_path=output_path)
            return 0 if ok else 1

        if self._args.monitor_on:
            self._route.monitor_on()
            print("Local monitoring is on.")
            return 0

        if self._args.monitor_off:
            self._route.monitor_off()
            print("Local monitoring is off.")
            return 0

        utterances = list(self._read_utterances())
        repl_enabled = self._should_start_repl()

        if not utterances and not repl_enabled:
            raise SSpeakError("No text was provided.")

        if self._args.monitor:
            self._route.monitor_on()

        sink = self._create_sink()
        previous_default_source = None

        if (
            isinstance(sink, PulseVirtualMicSink)
            and self._args.cleanup_on_exit
            and self._args.restore_default_source
            and self._route.is_available()
        ):
            previous_default_source = self._route.get_default_source()

        synthesizer = KokoroSynthesizer(
            lang_code=self._args.lang,
            voice=self._args.voice,
            speed=self._args.speed,
            model_config=Path(self._args.model_config).expanduser(),
            model_file=Path(self._args.model_file).expanduser(),
            repo_id=self._args.repo_id,
            voice_dir=Path(self._args.voice_dir).expanduser(),
            offline=not self._args.online,
            show_warnings=self._args.show_warnings,
        )

        sink.open()

        if isinstance(sink, PulseVirtualMicSink) and self._args.set_default_source:
            self._route.set_default_source()
            print(
                f"Default input source set to {self._args.mic_source}.",
                file=sys.stderr,
            )

            if self._args.move_recording_streams:
                moved = self._route.move_recording_streams_to_mic()

                if moved:
                    print(
                        f"Moved {moved} active recording stream(s) to "
                        f"{self._args.mic_source}.",
                        file=sys.stderr,
                    )

        try:
            synthesizer.load()
            self._speak_utterances(synthesizer, sink, utterances)

            if repl_enabled:
                self._run_repl(synthesizer, sink)
        finally:
            try:
                sink.close()
            finally:
                if isinstance(sink, PulseVirtualMicSink) and self._args.cleanup_on_exit:
                    self._cleanup_pulse_route(previous_default_source)

        return 0

    def _cleanup_pulse_route(self, previous_default_source: Optional[str]) -> None:
        """Clean up the temporary Pulse/PipeWire route after this run."""
        if self._args.restore_default_source:
            try:
                restored = self._route.restore_default_source(previous_default_source)

                if restored:
                    print(f"Restored default input source to {restored}.", file=sys.stderr)
                else:
                    print(
                        "No fallback input source was found; default source was not restored.",
                        file=sys.stderr,
                    )
            except SSpeakError as exc:
                print(f"Could not restore default input source: {exc}", file=sys.stderr)

        try:
            self._route.monitor_off()
        except SSpeakError as exc:
            print(f"Could not disable local monitor: {exc}", file=sys.stderr)

        try:
            self._route.teardown()
            print("Cleaned up SSpeak virtual microphone route.", file=sys.stderr)
        except SSpeakError as exc:
            print(f"Could not tear down SSpeak route: {exc}", file=sys.stderr)

    def _should_start_repl(self) -> bool:
        """Return True when SSpeak should enter interactive prompt mode."""
        if self._args.no_repl:
            return False

        if self._args.repl:
            return True

        return not self._args.text and sys.stdin.isatty()

    def _speak_utterances(
        self,
        synthesizer: KokoroSynthesizer,
        sink: AudioSink,
        utterances: Iterable[str],
    ) -> None:
        """Synthesize and play a sequence of utterances."""
        for utterance in utterances:
            cleaned = utterance.strip()

            if not cleaned:
                continue

            print(f"Processing: {cleaned!r}", file=sys.stderr)
            audio = synthesizer.synthesize(cleaned)

            if audio.size == 0:
                print("No audio generated for utterance.", file=sys.stderr)
                continue

            peak = float(np.max(np.abs(audio)))
            print(
                f"Speaking: {cleaned!r} ({audio.size} samples, peak={peak:.4f})",
                file=sys.stderr,
            )
            sink.play(audio, self._args.sample_rate)

    def _run_repl(self, synthesizer: KokoroSynthesizer, sink: AudioSink) -> None:
        """Run the interactive SSpeak prompt."""
        print()
        print("SSpeak REPL is ready.")
        print(f"Microphone source: {self._args.mic_source}")
        print("Type text and press Enter to speak.")
        print("Commands: :help, :status, :monitor on, :monitor off, :quit")
        print("Tip: leave Discord input on Default, or select SSpeakMic directly.")
        print()

        while True:
            try:
                line = input("sspeak> ")
            except EOFError:
                print()
                return

            stripped = line.strip()

            if not stripped:
                continue

            if stripped in {":quit", ":exit", "/quit", "/exit", "quit", "exit"}:
                return

            if stripped in {":help", "/help", "help"}:
                self._print_repl_help()
                continue

            if stripped in {":status", "/status"}:
                self._route.print_status()
                continue

            if stripped in {":monitor on", "/monitor on"}:
                self._route.monitor_on()
                print("Local monitoring is on.")
                continue

            if stripped in {":monitor off", "/monitor off"}:
                self._route.monitor_off()
                print("Local monitoring is off.")
                continue

            self._speak_utterances(synthesizer, sink, [stripped])

    @staticmethod
    def _print_repl_help() -> None:
        """Print REPL help."""
        print(
            "SSpeak REPL commands:\n"
            "  :help         Show this help.\n"
            "  :status       Print PipeWire/Pulse route status.\n"
            "  :monitor on   Also hear SSpeak through local speakers.\n"
            "  :monitor off  Stop local speaker monitoring.\n"
            "  :quit         Exit.\n"
            "\n"
            "Anything else is spoken through the configured backend."
        )

    def _create_sink(self) -> AudioSink:
        backend = self._args.backend

        if backend == "auto":
            if self._route.is_available():
                backend = "pulse"
            else:
                backend = "sounddevice"
                print(
                    "Pulse/PipeWire tools are unavailable; falling back to "
                    "sounddevice speaker playback. This will not expose a mic.",
                    file=sys.stderr,
                )

        if backend == "pulse":
            return PulseVirtualMicSink(
                self._runner,
                self._route,
                setup_route=self._args.auto_setup_mic,
            )

        if backend == "sounddevice":
            return SoundDeviceSink(self._args.device)

        if backend == "file":
            return FileSink(Path(self._args.output_wav).expanduser())

        raise SSpeakError(f"Unknown backend: {backend}")

    def _read_utterances(self) -> Iterator[str]:
        if self._args.text:
            text = " ".join(self._args.text).strip()
            yield from text.splitlines()
            return

        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
            if text:
                yield from text.splitlines()
            return

        return

    def _list_sounddevice_devices(self) -> None:
        try:
            import sounddevice as sd  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise SSpeakError("sounddevice is not installed.") from exc

        print(sd.query_devices())

    @staticmethod
    def _print_install_hints() -> None:
        print(
            "Mint/Ubuntu:\n"
            "  sudo apt update\n"
            "  sudo apt install -y python3 python3-venv python3-pip "
            "espeak-ng pipewire pipewire-pulse wireplumber "
            "pulseaudio-utils pipewire-bin libsndfile1\n\n"
            "Arch:\n"
            "  sudo pacman -S --needed python python-pip espeak-ng "
            "pipewire pipewire-pulse wireplumber libpulse libsndfile\n\n"
            "Python venv:\n"
            "  python3 -m venv .venv\n"
            "  . .venv/bin/activate\n"
            "  python -m pip install --upgrade pip wheel setuptools\n"
            "  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
            "  python -m pip install 'kokoro>=0.9.4' 'misaki[en]' "
            "'soundfile>=0.12.1' 'numpy>=1.26' 'sounddevice>=0.4.6'\n"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Kokoro TTS into a PipeWire/Pulse virtual microphone.",
    )

    parser.add_argument(
        "text",
        nargs="*",
        help=(
            "Text to speak. If omitted from an interactive terminal, SSpeak "
            "sets up the mic and starts REPL mode by default."
        ),
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Force interactive REPL mode after any provided text is spoken.",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Disable default REPL mode when no text is provided.",
    )
    parser.add_argument(
        "--no-setup-mic",
        dest="auto_setup_mic",
        action="store_false",
        help=(
            "Do not automatically create/verify SSpeakSpeaker and SSpeakMic "
            "before speaking."
        ),
    )
    parser.add_argument(
        "--no-default-source",
        dest="set_default_source",
        action="store_false",
        help=(
            "Do not set SSpeakMic as the default system input before speaking. "
            "By default SSpeak sets it so apps using Default input just work."
        ),
    )
    parser.add_argument(
        "--no-move-recording-streams",
        dest="move_recording_streams",
        action="store_false",
        help=(
            "Do not move active recording streams to SSpeakMic. By default "
            "SSpeak tries to move already-open apps such as Discord."
        ),
    )
    parser.add_argument(
        "--cleanup-on-exit",
        action="store_true",
        help=(
            "Remove SSpeakSpeaker and SSpeakMic when the app exits. "
            "Default behavior leaves the mic alive for Discord/OBS reliability."
        ),
    )
    parser.add_argument(
        "--no-restore-default-source",
        dest="restore_default_source",
        action="store_false",
        help=(
            "When using --cleanup-on-exit, do not restore the previous/default "
            "real microphone before removing SSpeakMic."
        ),
    )
    parser.set_defaults(
        auto_setup_mic=True,
        set_default_source=True,
        move_recording_streams=True,
        restore_default_source=True,
    )

    parser.add_argument(
        "--backend",
        choices=("auto", "pulse", "sounddevice", "file"),
        default="auto",
        help="Audio backend. Default: auto.",
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Voice id or path to local .pt voice file. Default: af_heart.",
    )
    parser.add_argument(
        "--voice-dir",
        default=str(DEFAULT_VOICE_DIR),
        help=f"Directory containing local Kokoro .pt voices. Default: {DEFAULT_VOICE_DIR}",
    )
    parser.add_argument(
        "--lang",
        default="auto",
        help="Kokoro lang_code. Use auto to infer from voice. Default: auto.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier. Default: 1.0.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Kokoro output sample rate. Default: {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help=f"Local Kokoro config path. Default: {DEFAULT_MODEL_CONFIG}",
    )
    parser.add_argument(
        "--model-file",
        default=str(DEFAULT_MODEL_FILE),
        help=f"Local Kokoro model path. Default: {DEFAULT_MODEL_FILE}",
    )
    parser.add_argument(
        "--repo-id",
        default="hexgrad/Kokoro-82M",
        help="Kokoro repo_id. Used for library metadata; no download in offline mode.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hugging Face/network fallback. Default is offline-first.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show known harmless Torch/Kokoro warnings.",
    )

    parser.add_argument(
        "--speaker-sink",
        default=DEFAULT_SPEAKER_SINK,
        help=f"Virtual speaker sink name. Default: {DEFAULT_SPEAKER_SINK}",
    )
    parser.add_argument(
        "--mic-source",
        default=DEFAULT_MIC_SOURCE,
        help=f"Virtual mic source name. Default: {DEFAULT_MIC_SOURCE}",
    )
    parser.add_argument(
        "--setup-mic",
        action="store_true",
        help="Create/verify the PipeWire virtual microphone route.",
    )
    parser.add_argument(
        "--recreate-route",
        action="store_true",
        help="Force-remove and recreate SSpeak route during --setup-mic.",
    )
    parser.add_argument(
        "--teardown-mic",
        action="store_true",
        help="Remove SSpeak virtual microphone route.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print route status.",
    )
    parser.add_argument(
        "--test-route",
        action="store_true",
        help="Record a generated test tone from SSpeakMic and verify it is non-silent.",
    )
    parser.add_argument(
        "--test-output",
        default="sspeak-route-test.wav",
        help="Output WAV for --test-route. Default: sspeak-route-test.wav.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Also route SSpeak audio to local default speakers.",
    )
    parser.add_argument(
        "--monitor-on",
        action="store_true",
        help="Turn local monitoring on.",
    )
    parser.add_argument(
        "--monitor-off",
        action="store_true",
        help="Turn local monitoring off.",
    )

    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="sounddevice output name substring or index for --backend sounddevice.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List sounddevice devices and exit.",
    )
    parser.add_argument(
        "--output-wav",
        default="sspeak-output.wav",
        help="WAV file or directory for --backend file.",
    )

    parser.add_argument(
        "--install-hints",
        action="store_true",
        help="Print install commands for Mint and Arch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    return parser


def main() -> int:
    """Program entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return SSpeakApplication(args).run()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Reference: https://github.com/hexgrad/kokoro
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator

import numpy as np
import sounddevice as sd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kokoro TTS to a virtual sink (PipeWire) for use as a mic via monitor."
    )
    p.add_argument(
        "text",
        nargs="*",
        help="Words to speak. If omitted, read stdin until EOF.",
    )
    p.add_argument(
        "--voice",
        default="af_heart",
        help="Voice id (see kokoro README on GitHub). Default: af_heart",
    )
    p.add_argument(
        "--lang",
        default="a",
        metavar="CODE",
        help="KPipeline lang_code: a=American English, b=British, etc. Must match voice.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier.",
    )
    p.add_argument(
        "--device",
        "-d",
        default=None,
        help="Output device name substring or PortAudio index. Default: system default output.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print output devices and exit.",
    )
    return p.parse_args()


def _resolve_output_device(spec: str | None) -> int | str | None:
    if spec is None:
        return None
    if spec.isdigit():
        return int(spec)
    devices = sd.query_devices()
    matches: list[tuple[int, str]] = []
    for i, dev in enumerate(devices):
        name = dev["name"]
        if spec.lower() in name.lower():
            print(dev)
            matches.append((i, name))
    if not matches:
        print(f"No output device contains {spec!r}. Use --list-devices.", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print("Multiple matches; pick an index with -d N:\n", file=sys.stderr)
        for idx, n in matches:
            print(f"  {idx}: {n}", file=sys.stderr)
        sys.exit(1)
    idx, name = matches[0]
    print(f"Using output device [{idx}] {name}", file=sys.stderr)
    return idx


def _read_text(args: argparse.Namespace) -> Iterator[str]:
    if args.text:
        return " ".join(args.text).strip().splitlines()
    return iter(sys.stdin)


def _play_chunks(
        stream: sd.OutputStream,
    samples: Iterator[np.ndarray],
    
) -> None:
    for chunk in samples:
        if chunk is None or chunk.size == 0:
            continue
        audio = np.asarray(chunk, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        # Kokoro-82M outputs 24 kHz, but we opened the stream with 48 kHz. Resample by repeating each sample.
        audio = np.repeat(audio, 2)
        
        # Kokoro-82M outputs mono, but we opened the stream with 2 channels. Duplicate the mono signal to stereo.
        if stream.channels == 2 and audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        stream.write(audio)


def main() -> None:
    args = _parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    device = _resolve_output_device(args.device)
    text = _read_text(args)

    
    from kokoro import KPipeline, KModel
    
    print("Loading model...", file=sys.stderr)
    model = KModel(config="model/config.json", model="model/kokoro-v1_0.pth")
    pipeline = KPipeline(lang_code=args.lang, model=model, repo_id='hexgrad/Kokoro-82M')

    with sd.OutputStream(samplerate=48_000, device=device, channels=2) as stream:
        for line in text:
            print(f"Processing: {line!r}", file=sys.stderr)
            gen = pipeline(line, voice=args.voice, speed=args.speed)
            print(f"Speaking: {line!r}", file=sys.stderr)
            def _as_numpy_f32(audio: object) -> np.ndarray:
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                arr = np.asarray(audio, dtype=np.float32)
                return arr.reshape(-1)

            def audio_iter() -> Iterator[np.ndarray]:
                for graphemes, phonemes, audio in gen:
                    yield _as_numpy_f32(audio)

            # Kokoro-82M outputs 24 kHz (see upstream examples).
            _play_chunks(stream, audio_iter())
    sd.wait()

if __name__ == "__main__":
    main()

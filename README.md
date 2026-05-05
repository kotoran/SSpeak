# SSpeak

SSpeak is a small command-line Kokoro TTS app that speaks into a Linux virtual microphone.

It is designed for Linux Mint and Arch Linux using PipeWire/Pulse compatibility.

Default audio route:

```text
Kokoro TTS
  -> SSpeakSpeaker
  -> SSpeakSpeaker.monitor
  -> pw-link
  -> SSpeakMic
  -> Discord / OBS / Zoom / browser / game
```

`SSpeakMic` is the microphone you select in other apps.

## Important design notes

- SSpeak is offline-first by default.
- The app expects local Kokoro model and voice files.
- `sounddevice` is only a fallback speaker-output backend.
- `sounddevice` does not create or expose a microphone.
- The virtual microphone is created through PipeWire/Pulse tools: `pactl`, `paplay`, `parecord`, and `pw-link`.

## Repository layout

Expected layout:

```text
SSpeak/
├── README.md
├── build.sh
├── main.py
├── requirements.txt
├── .gitignore
├── .gitattributes
├── model/
│   ├── config.json
│   └── kokoro-v1_0.pth
└── voices/
    ├── af_heart.pt
    └── other Kokoro voice .pt files
```

The source package does not include model or voice files. Add them before running `build.sh`.

At minimum, SSpeak expects:

```text
model/config.json
model/kokoro-v1_0.pth
voices/af_heart.pt
```

## Install system dependencies

### Linux Mint / Ubuntu

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    espeak-ng \
    pipewire \
    pipewire-pulse \
    wireplumber \
    pulseaudio-utils \
    pipewire-bin \
    libsndfile1
```

### Arch Linux

```bash
sudo pacman -S --needed \
    python \
    python-pip \
    espeak-ng \
    pipewire \
    pipewire-pulse \
    wireplumber \
    libpulse \
    libsndfile
```

## Add local Kokoro files

Create the expected folders:

```bash
mkdir -p model voices
```

Put the local files here:

```text
model/config.json
model/kokoro-v1_0.pth
voices/af_heart.pt
```

Add any other voice `.pt` files you want under `voices/`.

For offline use, do not rely on Kokoro downloading voices by name. SSpeak loads `voices/<voice>.pt` locally and passes the voice tensor into Kokoro.

## Build the venv

From the repo root:

```bash
chmod +x build.sh
./build.sh
```

To also install system dependencies automatically:

```bash
./build.sh --system-deps
```

Activate the venv:

```bash
. .venv/bin/activate
```

## First route test

Run:

```bash
. .venv/bin/activate
python main.py --test-route
```

Expected success:

```text
OK: SSpeakMic captured audio.
```

This test bypasses Kokoro. It generates a tone, sends it to `SSpeakSpeaker`, records from `SSpeakMic`, and confirms the recording is not silent.

## Normal use

Run:

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py
```

By default, `python main.py` does the following:

```text
1. creates/verifies SSpeakSpeaker
2. creates/verifies SSpeakMic
3. links SSpeakSpeaker.monitor to SSpeakMic using pw-link
4. sets SSpeakMic as the default system input
5. tries to move active recording streams to SSpeakMic
6. loads Kokoro once
7. starts the REPL
```

In the REPL:

```text
sspeak> Hello, this is SSpeak.
```

Useful REPL commands:

```text
:help
:status
:monitor on
:monitor off
:quit
:exit
```

## Discord / OBS / Zoom setup

In the target app, set the microphone to:

```text
SSpeakMic
```

You can also leave the input device on:

```text
Default
```

because SSpeak sets `SSpeakMic` as the default input on startup.

For Discord testing, disable or reduce these while debugging:

```text
Echo Cancellation
Noise Suppression / Krisp
Automatic Gain Control
Automatic Input Sensitivity
Noise Gate
```

## One-shot mode

Speak once and exit:

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --no-repl \
    "This speaks once and exits."
```

## Local monitor

By default, SSpeak sends audio to the virtual microphone. You may not hear it locally.

To also hear SSpeak through your speakers:

```bash
python main.py --monitor
```

Inside the REPL:

```text
sspeak> :monitor on
```

Turn it off:

```text
sspeak> :monitor off
```

or:

```bash
python main.py --monitor-off
```

## Cleanup behavior

Default behavior leaves `SSpeakMic` alive after SSpeak exits. This is intentional because it is more reliable for Discord, OBS, and Zoom.

Temporary mode:

```bash
python main.py --cleanup-on-exit
```

With `--cleanup-on-exit`, SSpeak tries to:

```text
1. restore the previous/default real microphone
2. turn off local monitoring
3. remove SSpeakSpeaker
4. remove SSpeakMic
```

Disable default-source restoration during cleanup:

```bash
python main.py --cleanup-on-exit --no-restore-default-source
```

## Manual route commands

Create or verify route:

```bash
python main.py --setup-mic
```

Force rebuild route:

```bash
python main.py --setup-mic --recreate-route
```

Remove route:

```bash
python main.py --teardown-mic
```

Show route status:

```bash
python main.py --status
```

Test route:

```bash
python main.py --test-route
```

## Flags for manual control

Do not start the REPL when no text is provided:

```bash
python main.py --no-repl
```

Do not automatically create/verify the mic route:

```bash
python main.py --no-setup-mic
```

Do not set `SSpeakMic` as default input:

```bash
python main.py --no-default-source
```

Do not move already-open recording streams:

```bash
python main.py --no-move-recording-streams
```

Fully manual mode:

```bash
python main.py \
    --no-setup-mic \
    --no-default-source \
    --no-move-recording-streams
```

## File output backend

Write generated speech to WAV instead of playing it:

```bash
python main.py --backend file --output-wav output.wav --no-repl \
    "This writes a WAV file."
```

## Speaker-only fallback

Use `sounddevice` speaker playback only:

```bash
python main.py --backend sounddevice --no-repl \
    "This plays to speakers only and does not expose a mic."
```

List sounddevice devices:

```bash
python main.py --list-devices
```

## Offline behavior

SSpeak defaults to offline mode. These environment variables are recommended:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py
```

To intentionally allow Hugging Face/network fallback:

```bash
python main.py --online
```

Offline is recommended for predictable startup.

## Git LFS

The model and voice files are large. If using GitHub or GitLab, use Git LFS:

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
```

Then add model and voice files:

```bash
git add model/config.json model/kokoro-v1_0.pth voices/*.pt
```

## Troubleshooting

### `SSpeakMic` does not appear

```bash
python main.py --setup-mic --recreate-route
python main.py --status
```

Check tools:

```bash
command -v pactl paplay parecord pw-link
```

### Route test is silent

```bash
python main.py --teardown-mic
python main.py --setup-mic --recreate-route
python main.py --test-route
```

### Discord sees the mic but hears nothing

Run:

```bash
python main.py --test-route
```

If the route test passes, use `pavucontrol` and check the Recording tab while Discord mic test is active:

```bash
pavucontrol
```

Make sure Discord's recording stream is attached to `SSpeakMic`.

You can also force default input:

```bash
pactl set-default-source SSpeakMic
pactl set-source-mute SSpeakMic 0
pactl set-source-volume SSpeakMic 125%
```

### You hear local monitor but Discord does not

That means:

```text
SSpeak -> SSpeakSpeaker -> speakers
```

works, but this may not:

```text
SSpeakSpeaker.monitor -> SSpeakMic -> Discord
```

Run:

```bash
python main.py --test-route
```

If it passes, restart Discord or use `pavucontrol` to move Discord's recording stream to `SSpeakMic`.

### No local sound

By default, audio goes to the virtual mic only. Use:

```bash
python main.py --monitor
```

or inside the REPL:

```text
:monitor on
```

## Final repo checklist

Commit these:

```text
README.md
build.sh
main.py
requirements.txt
.gitignore
.gitattributes
model/config.json
model/kokoro-v1_0.pth
voices/*.pt
```

Do not commit:

```text
.venv/
__pycache__/
*.wav
main.py.before-*
install-*.sh
SSpeak-main-*.py
*.patch
*.zip
*.tar.gz
```

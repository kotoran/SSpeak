#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

INSTALL_SYSTEM_DEPS=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  ./build.sh [options]

Options:
  --system-deps    Install system dependencies with apt or pacman.
  --python PATH    Python executable to use. Default: python3.
  -h, --help       Show this help.

Examples:
  ./build.sh --system-deps
  ./build.sh
  . .venv/bin/activate
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --system-deps)
            INSTALL_SYSTEM_DEPS=1
            shift
            ;;
        --python)
            if [ "$#" -lt 2 ]; then
                echo "error: --python requires an argument" >&2
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

install_system_deps() {
    echo "==> Installing system dependencies"

    if command -v apt >/dev/null 2>&1; then
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
        return
    fi

    if command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --needed \
            python \
            python-pip \
            espeak-ng \
            pipewire \
            pipewire-pulse \
            wireplumber \
            libpulse \
            libsndfile
        return
    fi

    cat >&2 <<'EOF'
error: unsupported package manager.

Install these manually:
  python3
  python3-venv
  python3-pip
  espeak-ng
  pipewire
  pipewire-pulse
  wireplumber
  pactl/paplay/parecord
  pw-link
  libsndfile
EOF
    exit 1
}

check_required_files() {
    echo "==> Checking local offline model files"

    missing=0

    for path in \
        "main.py" \
        "requirements.txt" \
        "model/config.json" \
        "model/kokoro-v1_0.pth" \
        "voices/af_heart.pt"
    do
        if [ ! -f "$path" ]; then
            echo "missing: $path" >&2
            missing=1
        fi
    done

    if [ "$missing" -ne 0 ]; then
        cat >&2 <<'EOF'

error: required offline model/voice files are missing.

Expected:
  model/config.json
  model/kokoro-v1_0.pth
  voices/af_heart.pt

Add the Kokoro model file and at least one local voice .pt file before running build.
EOF
        exit 1
    fi
}

start_audio_services() {
    echo "==> Starting PipeWire user services when available"

    systemctl --user enable --now pipewire pipewire-pulse wireplumber >/dev/null 2>&1 || true

    if command -v pactl >/dev/null 2>&1; then
        pactl info | grep -E "Server Name|Default Sink|Default Source" || true
    fi
}

create_venv() {
    echo "==> Creating Python venv"

    if [ ! -d ".venv" ]; then
        "${PYTHON_BIN}" -m venv .venv
    fi

    # shellcheck disable=SC1091
    . .venv/bin/activate

    echo "==> Upgrading pip tooling"
    python -m pip install --upgrade pip wheel setuptools

    echo "==> Installing CPU PyTorch"
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch

    echo "==> Installing SSpeak Python requirements"
    python -m pip install --upgrade -r requirements.txt
}

verify_python_deps() {
    echo "==> Verifying Python imports"

    # shellcheck disable=SC1091
    . .venv/bin/activate

    python - <<'PY'
import kokoro
import misaki
import numpy
import soundfile

print("kokoro:", getattr(kokoro, "__version__", "unknown"))
print("Python deps OK")
PY
}

verify_app_syntax() {
    echo "==> Checking main.py syntax"

    # shellcheck disable=SC1091
    . .venv/bin/activate

    python - <<'PY'
from pathlib import Path
import ast

ast.parse(Path("main.py").read_text(encoding="utf-8"))
print("main.py syntax OK")
PY
}

if [ "$INSTALL_SYSTEM_DEPS" -eq 1 ]; then
    install_system_deps
fi

check_required_files
start_audio_services
create_venv
verify_python_deps
verify_app_syntax

cat <<'EOF'

Build complete.

Normal use:

  . .venv/bin/activate
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py

Route test:

  python main.py --test-route

One-shot:

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py --no-repl "Hello from SSpeak."

Discord/OBS/Zoom input device:

  SSpeakMic

EOF

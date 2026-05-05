if [ ! -d .venv ]; then
    python3.11 -m venv .venv
    . .venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
    pip install -r requirements.txt
else
    . .venv/bin/activate
fi
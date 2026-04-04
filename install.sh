#!/bin/bash
set -euo pipefail

# ── VoxTerm Installer ──────────────────────────────────────
# curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash

REPO="https://github.com/dmarzzz/VoxTerm.git"
INSTALL_DIR="$HOME/.local/share/voxterm"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/.venv"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${CYAN}▸${RESET} $1"; }
done_() { echo -e "${GREEN}✓${RESET} $1"; }
dim()   { echo -e "${DIM}  $1${RESET}"; }

echo ""
echo -e "${BOLD}VOXTERM${RESET} — local voice transcription"
echo -e "${DIM}everything runs on your machine, nothing leaves${RESET}"
echo ""

# ── Check Python ──────────────────────────────────────────
info "checking python..."

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.9+ required but not found."
    echo ""
    echo "   Install it with:"
    echo "     brew install python@3.12    (macOS)"
    echo "     sudo apt install python3    (Linux)"
    exit 1
fi

done_ "found $PYTHON ($($PYTHON --version 2>&1))"

# ── Clone or update ───────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "updating voxterm..."
    git -C "$INSTALL_DIR" pull --ff-only --quiet 2>/dev/null || true
    done_ "updated"
else
    info "downloading voxterm..."
    rm -rf "$INSTALL_DIR"
    git clone --depth 1 --quiet "$REPO" "$INSTALL_DIR"
    done_ "downloaded"
fi

# ── Create venv & install deps ────────────────────────────
info "setting up environment..."
dim "this may take a minute on first install"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --quiet --upgrade pip 2>/dev/null
"$VENV_DIR/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"

done_ "dependencies installed"

# ── Create launcher ───────────────────────────────────────
info "creating voxterm command..."

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/voxterm" << 'LAUNCHER'
#!/bin/bash
INSTALL_DIR="$HOME/.local/share/voxterm"
exec "$INSTALL_DIR/.venv/bin/python" -m tui.app "$@"
LAUNCHER
chmod +x "$BIN_DIR/voxterm"

done_ "installed to $BIN_DIR/voxterm"

# ── Check PATH ────────────────────────────────────────────
if ! echo "$PATH" | tr ':' '\n' | grep -q "$BIN_DIR"; then
    echo ""
    echo -e "${CYAN}▸${RESET} add this to your shell profile (~/.zshrc or ~/.bashrc):"
    echo ""
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "  then restart your terminal, or run:"
    echo ""
    echo "    source ~/.zshrc"
    echo ""
fi

# ── Done ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}voxterm installed!${RESET}"
echo ""
echo "  run it:     voxterm"
echo "  update:     curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash"
echo "  uninstall:  rm -rf ~/.local/share/voxterm ~/.local/bin/voxterm"
echo ""

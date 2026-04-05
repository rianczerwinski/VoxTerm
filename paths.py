"""Centralized, platform-aware path resolution for VoxTerm.

On macOS (darwin): uses ~/Documents/voxterm and ~/Library/Application Support/voxterm.
On Linux: uses XDG Base Directory paths ($XDG_DATA_HOME, $XDG_CONFIG_HOME).
"""

import os
import sys
from pathlib import Path

_home = Path.home()

if sys.platform == "darwin":
    SESSIONS_DIR = _home / "Documents" / "voxterm-transcripts"
    DATA_DIR = _home / "Library" / "Application Support" / "voxterm"
    LIVE_DIR = SESSIONS_DIR / ".live"
    BIN_DIR = DATA_DIR / ".bin"
    CRASH_DIR = DATA_DIR / ".crashes"
    STATE_FILE = DATA_DIR / ".state.json"
elif sys.platform.startswith("linux"):
    # Linux — XDG-compliant paths
    _xdg_data = Path(os.environ.get("XDG_DATA_HOME", _home / ".local" / "share"))
    _xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", _home / ".config"))
    DATA_DIR = _xdg_data / "voxterm"
    CONFIG_DIR = _xdg_config / "voxterm"
    SESSIONS_DIR = DATA_DIR
    LIVE_DIR = DATA_DIR / ".live"
    BIN_DIR = DATA_DIR / ".bin"
    CRASH_DIR = DATA_DIR / ".crashes"
    STATE_FILE = CONFIG_DIR / "state.json"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

# Speaker database
DB_DIR = DATA_DIR
DB_PATH = DB_DIR / ".speakers.db"
BACKUP_DIR = DB_DIR / ".backups"

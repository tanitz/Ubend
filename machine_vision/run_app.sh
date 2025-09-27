#!/usr/bin/env bash
# ...existing code...
WORKDIR="/home/snc/Documents/Ubend/machine_vision"
VENV="$WORKDIR/ubend/bin/activate"
LOG="$WORKDIR/run_app.log"

cd "$WORKDIR" || { echo "Cannot cd $WORKDIR"; exit 1; }

if [ -f "$VENV" ]; then
    # shellcheck source=/dev/null
    . "$VENV"
fi

# echo "=== $(date) ===" >> "$LOG"
# echo "PWD=$(pwd)" >> "$LOG"
# echo "PYTHON=$(which python)" >> "$LOG"

# รันแอป (stdout/stderr => log) — ไม่ใช้ exec เพื่อให้สคริปต์คืนมาควบคุม
python main.py
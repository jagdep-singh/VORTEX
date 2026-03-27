#!/bin/sh
set -eu

DEFAULT_CWD="/workspace"

has_cwd_flag=0
for arg in "$@"; do
    if [ "$arg" = "--cwd" ] || [ "$arg" = "-c" ]; then
        has_cwd_flag=1
        break
    fi
done

if [ "$#" -eq 0 ]; then
    exec python /app/main.py --cwd "$DEFAULT_CWD"
fi

if [ "$has_cwd_flag" -eq 1 ]; then
    exec python /app/main.py "$@"
fi

exec python /app/main.py --cwd "$DEFAULT_CWD" "$@"

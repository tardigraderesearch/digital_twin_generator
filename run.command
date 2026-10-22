#!/bin/sh
# macOS: double-click this file in Finder to start Digital Twin Generator.
cd "$(dirname "$0")" || exit 1
./run.sh "$@"
status=$?
if [ "$status" -ne 0 ]; then
    printf 'Press Return to close this window.'
    read -r _
fi
exit "$status"

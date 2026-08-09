#!/bin/bash
# Run a command, killing it if total julia RSS exceeds LIMIT_GB.  Guards the
# machine against a repeat of the OOM that took the session down.
LIMIT_GB=${LIMIT_GB:-40}
"$@" & CMD=$!
while kill -0 $CMD 2>/dev/null; do
  RSS=$(ps -o rss= -C julia 2>/dev/null | awk '{s+=$1} END {print int(s/1000000)}')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$LIMIT_GB" ]; then
    echo "WATCHDOG: julia RSS ${RSS} GB > ${LIMIT_GB} GB, killing" >&2
    pkill -9 -f "carina.jl" ; kill -9 $CMD 2>/dev/null; exit 99
  fi
  sleep 5
done
wait $CMD

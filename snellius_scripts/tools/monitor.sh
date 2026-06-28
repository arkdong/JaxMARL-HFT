#!/usr/bin/env bash
set -euo pipefail
cd /home/adong/JaxMARL-HFT
source snellius_scripts/00_cluster.sh
case "${1:-queue}" in
  queue)
    squeue -u "$USER"
    ;;
  tail)
    log="${2:-}"
    if [[ -z "$log" ]]; then
      echo "Usage: bash snellius_scripts/tools/monitor.sh tail <log-file>" >&2
      echo "Recent logs:" >&2
      ls -1t "$LOG_DIR" | head -20 >&2 || true
      exit 1
    fi
    tail -f "$log"
    ;;
  latest)
    latest="$(ls -1t "$LOG_DIR"/*.out 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] || { echo "No .out logs found in $LOG_DIR" >&2; exit 1; }
    echo "$latest"
    tail -f "$latest"
    ;;
  acct)
    jobid="${2:-}"
    [[ -n "$jobid" ]] || { echo "Usage: bash snellius_scripts/tools/monitor.sh acct <JOBID>" >&2; exit 1; }
    sacct -j "$jobid" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
    ;;
  *)
    echo "Usage: $0 {queue|latest|tail <file>|acct <jobid>}" >&2
    exit 1
    ;;
esac

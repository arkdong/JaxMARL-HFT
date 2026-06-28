#!/usr/bin/env bash
set -euo pipefail
# Usage examples:
#   bash 07_monitoring_commands.sh queue
#   bash 07_monitoring_commands.sh tail jmhft-smoke <JOBID>
#   bash 07_monitoring_commands.sh acct <JOBID>

cmd="${1:-help}"
case "${cmd}" in
  queue)
    squeue -u "${USER}"
    ;;
  tail)
    jobname="${2:?give job name, e.g. jmhft-smoke}"
    jobid="${3:?give job id}"
    tail -f "/home/adong/JaxMARL-HFT/slurm_logs/${jobname}-${jobid}.out"
    ;;
  acct)
    jobid="${2:?give job id}"
    sacct -j "${jobid}" --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
    ;;
  cancel)
    jobid="${2:?give job id}"
    scancel "${jobid}"
    ;;
  *)
    cat <<EOF
Commands:
  bash 07_monitoring_commands.sh queue
  bash 07_monitoring_commands.sh tail jmhft-smoke <JOBID>
  bash 07_monitoring_commands.sh tail jmhft-train <JOBID>
  bash 07_monitoring_commands.sh acct <JOBID>
  bash 07_monitoring_commands.sh cancel <JOBID>
EOF
    ;;
esac

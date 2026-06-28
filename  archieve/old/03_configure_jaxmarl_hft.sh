#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: venv Python not found: ${VENV_DIR}/bin/python" >&2
  echo "Run 02_setup_python_env.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

"${VENV_DIR}/bin/python" - <<'PY'
import json
import os
import re
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"]).resolve()
data = Path(os.environ["DATA_ROOT"]).resolve()
stock = os.environ["STOCK"]
period = os.environ["PERIOD"]

env_path = repo / "config" / "env_configs" / "2_player_fq_fqc.json"
rl_path = repo / "config" / "rl_configs" / "ippo_rnn_JAXMARL_2player.yaml"
train_py = repo / "gymnax_exchange" / "jaxrl" / "MARL" / "ippo_rnn_JAXMARL.py"

for path in (env_path, rl_path, train_py):
    if not path.exists():
        raise FileNotFoundError(path)
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text())

cfg = json.loads(env_path.read_text())
cfg["world_config"].update({
    "alphatradePath": str(repo),
    "dataPath": str(data),
    "stock": stock,
    "timePeriod": period,
})
env_path.write_text(json.dumps(cfg, indent=2) + "\n")

text = rl_path.read_text()
text = re.sub(r'(?m)^"TimePeriod"\s*:.*$', f'"TimePeriod" : "{period}"', text)
text = re.sub(r'(?m)^"EvalTimePeriod"\s*:.*$', f'"EvalTimePeriod" : "{period}"', text)
rl_path.write_text(text)

# Make JAX memory settings override-able from the sbatch script.
# The original file assigns os.environ[...] before importing JAX.
source = train_py.read_text()
source = source.replace(
    'os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"',
    'os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")',
)
source = source.replace(
    'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"',
    'os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")',
)
train_py.write_text(source)

lobster_dir = data / "rawLOBSTER" / stock / period
print("Configured JaxMARL-HFT:")
print(f"  env config: {env_path}")
print(f"  RL config:  {rl_path}")
print(f"  repo root:  {repo}")
print(f"  data root:  {data}")
print(f"  dataset:    {lobster_dir}")
print(f"  files seen: {len(list(lobster_dir.glob('*'))) if lobster_dir.exists() else 0}")
PY

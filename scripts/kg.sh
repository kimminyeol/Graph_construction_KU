#!/usr/bin/env bash
# Detached pipeline runner: keeps running after VSCode / the terminal / SSH closes.
#
#   scripts/kg.sh start  configs/kg.yaml [--from verify | --steps build export ...]
#   scripts/kg.sh status [configs/kg.yaml]     # all runs if omitted
#   scripts/kg.sh log    configs/kg.yaml       # follow the latest log (Ctrl-C only stops viewing)
#   scripts/kg.sh stop   configs/kg.yaml       # graceful; `start` again resumes collection
#
# Python: $KG_PYTHON, default = the kg_construction conda env (no `conda activate` needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${KG_PYTHON:-python}"
LOGS="$ROOT/data/logs"
cd "$ROOT"
# Only GPU 0 may ever be used (GPU 1 is not ours). The code pins it too; this is the outer guard.
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1   # the xet downloader stalled on this host; plain HTTP works

run_name() { "$PY" -c "from kg.config import load_config; print(load_config('$1').run_name)"; }
status_file() { echo "$LOGS/$1.status.json"; }
field() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$1" "$2"; }
alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }

show_status() {
  local f="$1" pid state
  pid=$(field "$f" pid); state=$(field "$f" state)
  if [[ "$state" == "running" ]] && ! alive "$pid"; then state="dead (killed without cleanup; start again to resume)"; fi
  "$PY" - "$f" "$state" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); state = sys.argv[2]
print(f"[{s['run_name']}] {state}  pid={s['pid']}  started={s['started_at']}  ended={s.get('ended_at', '-')}")
print(f"  steps: " + " ".join(f"{n}({s['done_steps'][n]}s)" if n in s['done_steps']
                              else (f"*{n}*" if n == s.get('current_step') else n) for n in s['steps']))
for k in ("warning", "error"):
    if s.get(k):
        print(f"  {k}: {s[k]}")
EOF
}

cmd="${1:-}"; shift || true
case "$cmd" in
  start)
    cfg="${1:?usage: kg.sh start <config> [pipeline args]}"; shift
    name=$(run_name "$cfg"); f=$(status_file "$name")
    if [[ -f "$f" ]] && [[ "$(field "$f" state)" == "running" ]] && alive "$(field "$f" pid)"; then
      echo "run '$name' is already running (pid $(field "$f" pid))"; exit 1
    fi
    mkdir -p "$LOGS"
    out="$LOGS/${name}.nohup.out"
    # setsid: own session, so closing the terminal/VSCode sends it no SIGHUP; nohup: belt and braces
    nohup setsid "$PY" -m kg.pipeline --config "$cfg" "$@" >"$out" 2>&1 </dev/null &
    sleep 2
    if [[ -f "$f" ]] && alive "$(field "$f" pid)"; then
      echo "started '$name' (pid $(field "$f" pid)); stdout: $out"
      echo "check: scripts/kg.sh status $cfg   |   follow: scripts/kg.sh log $cfg"
    else
      echo "failed to start; see $out"; tail -20 "$out"; exit 1
    fi
    ;;
  status)
    if [[ $# -ge 1 ]]; then show_status "$(status_file "$(run_name "$1")")"
    else
      shopt -s nullglob; files=("$LOGS"/*.status.json)
      [[ ${#files[@]} -eq 0 ]] && { echo "no runs yet"; exit 0; }
      for f in "${files[@]}"; do show_status "$f"; done
    fi
    ;;
  log)
    name=$(run_name "${1:?usage: kg.sh log <config>}")
    latest=$(ls -t "$LOGS/${name}_"*.log 2>/dev/null | head -1)
    [[ -z "$latest" ]] && { echo "no log for $name"; exit 1; }
    echo "== $latest"; tail -n 30 -f "$latest"
    ;;
  stop)
    name=$(run_name "${1:?usage: kg.sh stop <config>}"); f=$(status_file "$name"); pid=$(field "$f" pid)
    if alive "$pid"; then kill -TERM "$pid"; echo "sent SIGTERM to $pid ($name)"; else echo "'$name' is not running"; fi
    ;;
  *)
    sed -n '2,9p' "$0"; exit 1 ;;
esac

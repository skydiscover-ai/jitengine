#!/bin/bash
# Install a JITEngine engine and the agent launcher on this Mac, in one command:
#
#   curl -fsSL https://jitengine.ai/install.sh | bash -s -- flint   # or qwen3.6, flash-next
#
# It picks a build for this Mac's chip and memory (see the choice below), installs it and
# jit-launch into ~/.jitengine, downloads the model at its pinned revision into ~/models (for Flash-Next, also packs
# its experts for streaming, once), and adds a `jitengine` command: on the PATH of new terminals through the shell
# profile, and of this one through a link in the first of ~/.local/bin, ~/bin, /opt/homebrew/bin and /usr/local/bin
# that is on it and writable. Then `jitengine claude` (or codex, opencode, pi) starts the engine if it is not running,
# waits until it is ready, and starts the agent on it.
#
# Environment: JITENGINE_HOME (default ~/.jitengine), JITENGINE_MODELS (default ~/models), JITENGINE_REPO and
# JITENGINE_REF (the repository and branch to fetch the engines from when this script does not run from a checkout;
# by default the public repository's default branch), and
# JITENGINE_NO_MODIFY_PATH=1 to leave the shell profile and the PATH folders alone.
set -euo pipefail

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'jitengine install: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: curl -fsSL https://jitengine.ai/install.sh | bash -s -- [flint | qwen3.6 | flash-next]   (default: flint)

Installs an engine build chosen for this Mac's chip and memory and jit-launch into ~/.jitengine, downloads the model
into ~/models, and adds a `jitengine` command: `jitengine start` serves the model on http://127.0.0.1:8080, and
`jitengine claude` (or codex, opencode, pi) starts the engine if it is not running and the agent on it.

Environment: JITENGINE_HOME, JITENGINE_MODELS, JITENGINE_REPO, JITENGINE_REF, and JITENGINE_NO_MODIFY_PATH=1 to
leave the shell profile and the PATH folders alone.
EOF
    exit 0 ;;
esac

MODEL="${1:-flint}"
HOME_DIR="${JITENGINE_HOME:-$HOME/.jitengine}"
MODELS="${JITENGINE_MODELS:-$HOME/models}"
REPO="${JITENGINE_REPO:-https://github.com/skydiscover-ai/jitengine.git}"
REF="${JITENGINE_REF:-}"

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this needs a Mac with Apple silicon"
MEM=$(sysctl -n hw.memsize); G=$((MEM / 1073741824))
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)   # e.g. "Apple M2", "Apple M4 Pro"
# The build, from what our two Macs measured (budget sweep, 2026-11-03). Qwen3.6: the M4 Pro build from 32 GB, where its 4-bit
# model fits in memory, and the M2 build below. Flint: in memory the two builds decode alike, and the M2 build streams
# faster when memory is short, so the M2 build below 40 GB. Flash-Next: the M2 build on a base M1 or M2 (faster on our
# M2), the M4 Pro build on other chips (faster on our M4 Pro at every budget).
case "$MODEL" in
  flash-next|flashnext) case "$CHIP" in "Apple M1"|"Apple M2") MAC=m2 ;; *) MAC=m4pro ;; esac ;;
  qwen3.6) if [ "$G" -ge 32 ]; then MAC=m4pro; else MAC=m2; fi ;;
  *) if [ "$G" -ge 40 ]; then MAC=m4pro; else MAC=m2; fi ;;
esac

WIRED_MB=""
ARGS=""
NEED_GIB=18
RESIDENT_GIB=18
EXTRA_REPO=""
BUDGET_ARGS=""   # the budget to pass when it fits in the memory available ("": the engine's default)
PACK_VAR=""      # Flash-Next: the variable that tells the engine where its packs are
PACK_ARGS=""
PACK_GB=0
case "$MODEL:$MAC" in
  flint:*)
    ENGINE=flint-$MAC
    REPO_ID=RecursiveSI/flint-int4; REV=3b93f9053b530488405ae9fca51d4900d6ce21f0; DIR=flint-int4
    NEED_GIB=31; RESIDENT_GIB=26   # 31 GiB holds the model and 261K tokens of history (262K needs 31.02), 26 GiB 73K
    if [ "$MAC" = m2 ]; then
      # From 24 GB, memory less the 6 GiB the engines leave macOS, with the GPU wired limit raised to memory less 2 GiB:
      # 18 GiB on a 24 GB Mac, where the M2 build was measured (limit 22 GiB), 26 on 32 GB and 30 on 36 GB, which hold
      # the whole model and 73K or more tokens of history. Below 24 GB, the limit macOS gives (2/3 of memory), unraised.
      if [ "$G" -ge 24 ]; then W=$((G - 6)); WIRED_MB=$(((G - 2) * 1024)); else W=$((G * 2 / 3)); fi
      NEED_GIB=$W; RESIDENT_GIB=25
    fi ;;
  qwen3.6:m2)
    ENGINE=qwen3.6-35b-a3b-m2
    REPO_ID=andrevp/Qwen3.6-35B-A3B-3bit-MLX; REV=794408c4b69a90c9a7b96d3fce6949cccf08df85; DIR=Qwen3.6-35B-A3B-3bit
    ARGS="--max-context 32768 --prefix-cache-mib 1024 --build-expert-pack"; NEED_GIB=18; RESIDENT_GIB=22
    # This agent setting (32K context, 1 GiB prefix cache) keeps the model in memory from 22 GiB (2026-11-03 sweep),
    # more than a 24 GB Mac's 18; below 24 GB and from 26 GB the budget is memory less 6 GiB, at most 34 (10 GiB on a
    # 16 GB Mac, the smallest budget this setting ran at; at 8 GiB it refuses to start).
    if [ "$G" -lt 24 ] || [ "$G" -ge 26 ]; then
      NEED_GIB=$((G - 6)); [ "$NEED_GIB" -le 34 ] || NEED_GIB=34; BUDGET_ARGS="--memory-budget-gib $NEED_GIB"
    fi ;;
  qwen3.6:m4pro)
    ENGINE=qwen3.6-35b-a3b-m4pro
    REPO_ID=mlx-community/Qwen3.6-35B-A3B-4bit; REV=38740b847e4cb78f352aba30aa41c76e08e6eb46; DIR=Qwen3.6-35B-A3B-4bit
    EXTRA_REPO="mlx-community/Qwen3.6-35B-A3B-MTP-4bit 0295b81421bf4d0fccca9a7c0fcfb1418dda3516 Qwen3.6-35B-A3B-MTP-4bit"
    ARGS="--max-context 32768 --build-expert-pack"; NEED_GIB=34; RESIDENT_GIB=26   # in memory from 26 GiB (2026-11-03 sweep)
    if [ "$G" -lt 40 ]; then
      # 32 and 36 GB: memory less 6 GiB (26 and 30), with the GPU wired limit raised to memory less 2 GiB. Our M4 Pro with
      # its limit set to two-thirds of 32 GiB (21.3 GiB) streamed experts, 20.1 tok/s, against 166.4 in memory with it
      # raised (2026-11-03); no real 32 or 36 GB Mac's default limit was measured, so both get the raise.
      NEED_GIB=$((G - 6)); BUDGET_ARGS="--memory-budget-gib $NEED_GIB"; WIRED_MB=$(((G - 2) * 1024))
    fi ;;
  flash-next:*|flashnext:*)
    [ "$G" -ge 24 ] || die "Flash-Next needs a Mac with 24 GB of memory or more"
    ENGINE=qwen3.8-flash-next-$MAC
    REPO_ID=mlx-community/Qwen3.8-Flash-Next-4bit; REV=07b5dc6c54600a359b87f1e53e7adf6351c72a2c; DIR=Qwen3.8-Flash-Next-4bit
    # The 112 GB model streams from packs built once: 108 GB, or 178 GB for the M4 Pro build with its MTP head and
    # pack16 (every measured M4 Pro row used both).
    PACK_VAR="JITENGINE_QWEN3_8_FLASH_NEXT_$(echo "$MAC" | tr a-z A-Z)_PACK_DIR"; PACK_GB=108
    if [ "$MAC" = m4pro ]; then PACK_ARGS="--mtp --pack16"; PACK_GB=178; fi
    # The budget: the measured 18 GB on a 24 GB Mac and 40 GB on a 48 GB one, else 78% of memory up to 40 GB, the
    # largest budget measured (the engines refuse more than 80% without --force).
    case $G in
      24) GB=18 ;;
      48) GB=40 ;;
      *) GB=$((MEM * 78 / 100 / 1000000000)); [ "$GB" -le 40 ] || GB=40 ;;
    esac
    BUDGET_ARGS="--memory-budget-gb $GB"; NEED_GIB=$((GB * 1000000000 / 1073741824)); RESIDENT_GIB=$((NEED_GIB + 1)) ;;
  *) die "unknown model '$MODEL': use flint, qwen3.6 or flash-next" ;;
esac
say "This Mac (${CHIP:-Apple silicon}, $G GB) gets the $ENGINE engine"

PY=""
for c in python3.14 /opt/homebrew/bin/python3.14 /usr/local/bin/python3.14; do
  if command -v "$c" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
done
if [ -z "$PY" ]; then
  BREW=$(command -v brew || true)
  [ -n "$BREW" ] || [ ! -x /opt/homebrew/bin/brew ] || BREW=/opt/homebrew/bin/brew
  [ -n "$BREW" ] || die "this needs Python 3.14: install Homebrew from https://brew.sh, then run this again"
  say "Installing Python 3.14 with Homebrew"
  "$BREW" install python@3.14
  PY="$("$BREW" --prefix python@3.14)/bin/python3.14"
fi

SELF="${BASH_SOURCE[0]:-}"
if [ -n "$SELF" ] && [ -f "$SELF" ] && [ -d "$(dirname "$SELF")/engines/$ENGINE" ]; then
  SRC=$(cd "$(dirname "$SELF")" && pwd)
else
  SRC="$HOME_DIR/src"
  if [ -d "$SRC/.git" ]; then
    say "Updating the engines in $SRC"
    git -C "$SRC" fetch -q --depth 1 origin "${REF:-HEAD}" && git -C "$SRC" checkout -q FETCH_HEAD
  else
    say "Fetching the engines into $SRC"
    git clone -q --depth 1 ${REF:+--branch "$REF"} "$REPO" "$SRC"
  fi
fi

say "Installing $ENGINE and jit-launch into $HOME_DIR"
mkdir -p "$HOME_DIR/bin" "$MODELS"
[ -x "$HOME_DIR/venv/bin/python" ] || "$PY" -m venv "$HOME_DIR/venv"
"$HOME_DIR/venv/bin/pip" install -q --disable-pip-version-check "$SRC/engines/$ENGINE[download]" "$SRC/launch"

download() {
  say "Downloading $1 into $MODELS/$3 (resumes if interrupted)"
  "$HOME_DIR/venv/bin/hf" download "$1" --revision "$2" --local-dir "$MODELS/$3" >/dev/null
}
if [ -n "$PACK_VAR" ]; then
  PACK_DIR="$MODELS/flash-next-pack"
  if ! ls "$PACK_DIR/$DIR"-*.experts >/dev/null 2>&1; then
    NEED_GB=$((PACK_GB + 4))
    [ -f "$MODELS/$DIR/model.safetensors.index.json" ] || NEED_GB=$((NEED_GB + 112))
    FREE_GB=$(df -k "$MODELS" | awk 'NR == 2 {printf "%d", $4 * 1024 / 1e9}')
    [ "$FREE_GB" -ge "$NEED_GB" ] || die "Flash-Next needs about $NEED_GB GB free on the disk holding $MODELS; it has $FREE_GB GB"
  fi
fi
download "$REPO_ID" "$REV" "$DIR"
[ -z "$EXTRA_REPO" ] || download $EXTRA_REPO
if [ -n "$PACK_VAR" ]; then
  say "Packing the experts for streaming into $PACK_DIR (once, about $PACK_GB GB; a later run only checks it)"
  mkdir -p "$PACK_DIR"
  env "$PACK_VAR=$PACK_DIR" "$HOME_DIR/venv/bin/jitengine-$ENGINE" pack --model-path "$MODELS/$DIR" $PACK_ARGS
fi

cat > "$HOME_DIR/engine.env" <<EOF
ENGINE_CMD=jitengine-$ENGINE
MODEL_PATH="$MODELS/$DIR"
SERVE_ARGS="$ARGS"
WIRED_MB="$WIRED_MB"
NEED_GIB=$NEED_GIB
RESIDENT_GIB=$RESIDENT_GIB
BUDGET_ARGS="$BUDGET_ARGS"
EOF
[ -z "$PACK_VAR" ] || echo "export $PACK_VAR=\"$PACK_DIR\"" >> "$HOME_DIR/engine.env"

cat > "$HOME_DIR/bin/jitengine" <<'EOF'
#!/bin/bash
# jitengine: run a coding agent on the local engine, starting the engine first if it is not running.
#   jitengine claude | codex | opencode | pi [-- agent args]   start the engine if needed, then the agent on it
#   jitengine serve [serve args]                               run the engine in this terminal
#   jitengine start | restart | stop | status | logs
# If the engine's default memory budget does not fit in the memory available now, start passes the largest budget
# that does (the engine then reads more from the SSD and runs slower); JITENGINE_BUDGET_GIB=N sets one yourself.
set -euo pipefail
HOME_DIR="${JITENGINE_HOME:-$HOME/.jitengine}"
. "$HOME_DIR/engine.env"
BIN="$HOME_DIR/venv/bin"
URL=http://127.0.0.1:8080
LOG="$HOME_DIR/serve.log"
PID="$HOME_DIR/serve.pid"

up() { curl -fsS -m 2 "$URL/v1/models" >/dev/null 2>&1; }
wired() {
  [ -n "$WIRED_MB" ] || return 0
  local now; now=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
  if [ "$now" -lt "$WIRED_MB" ]; then
    echo "This engine needs the GPU wired limit at $WIRED_MB MB (now $now). Raising it until the next restart:"
    sudo sysctl "iogpu.wired_limit_mb=$WIRED_MB" >/dev/null
  fi
}
available_gib() {  # what macOS can hand out without paging out app memory, as the engines' start check counts it:
  # all but wired memory, the compressor and app (anonymous) pages; cached files count, since macOS drops them
  vm_stat | awk -v total="$(sysctl -n hw.memsize)" '/page size of/ {p = $8}
    /^Pages wired down:/ {w = $NF} /^Pages occupied by compressor:/ {c = $NF} /^Pages purgeable:/ {g = $NF}
    /^Anonymous pages:/ {a = $NF} /^Pages (free|inactive|speculative|purgeable):/ {n += $NF}
    END {if (a == "") printf "%.1f", n * p / 1073741824; else printf "%.1f", (total - (w + c + a - g) * p) / 1073741824}'
}
budget_args() {
  if [ -n "${JITENGINE_BUDGET_GIB:-}" ]; then echo "--memory-budget-gib $JITENGINE_BUDGET_GIB"; return; fi
  local avail fit; avail=$(available_gib)
  fit=$(awk -v a="$avail" 'BEGIN {printf "%d", a - 1.5}')   # leave 1.5 GiB for the agent and the rest of the Mac
  if [ "$fit" -lt "$NEED_GIB" ]; then
    echo "Memory: $avail GiB available now, so the engine gets a $fit GiB budget instead of $NEED_GIB GiB." >&2
    if [ "$fit" -ge "$RESIDENT_GIB" ]; then
      echo "The whole model still fits, with a shorter context limit; close apps and run 'jitengine restart' for more." >&2
    else
      echo "It reads more from the SSD and runs slower; close apps and run 'jitengine restart' for full speed." >&2
    fi
    echo "--memory-budget-gib $fit"
  else
    echo "${BUDGET_ARGS:-}"
  fi
}
start() {
  up && return 0
  wired
  local budget; budget=$(budget_args)
  echo "Starting $ENGINE_CMD in the background (log: ${LOG/#$HOME/~})"
  # ignore hangups, as nohup does, without nohup's need for a terminal
  (trap '' HUP; exec "$BIN/$ENGINE_CMD" serve --model-path "$MODEL_PATH" $SERVE_ARGS $budget) > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PID"
  local i=0
  until up; do
    if ! kill -0 "$(cat "$PID")" 2>/dev/null; then echo "The engine stopped. The end of $LOG:"; tail -20 "$LOG"; exit 1; fi
    i=$((i + 1))
    [ $((i % 10)) -ne 0 ] || echo "  loading, $((i * 2)) s $(tail -1 "$LOG" | cut -c1-90)"
    sleep 2
  done
  echo "The engine is ready on $URL"
}
case "${1:-}" in
  serve)
    shift; wired
    case " $* " in *" --memory-budget"*) budget="" ;; *) budget=$(budget_args) ;; esac   # a budget you pass wins
    exec "$BIN/$ENGINE_CMD" serve --model-path "$MODEL_PATH" $SERVE_ARGS $budget "$@" ;;
  start) start ;;
  restart) "$0" stop; sleep 2; start ;;
  stop)
    if [ -s "$PID" ] && kill "$(cat "$PID")" 2>/dev/null; then echo "Stopped the engine"; else echo "No engine started by jitengine is running"; fi
    : > "$PID" ;;
  status) if up; then echo "Running on $URL ($ENGINE_CMD)"; else echo "Not running"; fi ;;
  logs) exec tail -f "$LOG" ;;
  ""|-h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) start; exec "$BIN/jit-launch" "$@" ;;
esac
EOF
chmod +x "$HOME_DIR/bin/jitengine"

CMD="$HOME_DIR/bin/jitengine"
LINE="export PATH=\"$HOME_DIR/bin:\$PATH\""
PROFILE="$HOME/.zprofile"; [ "${SHELL##*/}" != bash ] || PROFILE="$HOME/.bash_profile"   # zsh is macOS's default
ours() { [ "$1" = "$CMD" ] || [ "$(readlink "$1" 2>/dev/null)" = "$CMD" ]; }
if [ "${JITENGINE_NO_MODIFY_PATH:-}" != 1 ]; then
  grep -qsF "$LINE" "$PROFILE" || printf '\n# JITEngine\n%s\n' "$LINE" >> "$PROFILE"
  # The profile reaches new terminals only; the one that ran this keeps its PATH (this script has the same one). A link
  # in a folder already on it, the user's own or Homebrew's, makes the next command, `jitengine claude`, work there too.
  if ! ours "$(command -v jitengine || true)"; then
    for d in "$HOME/.local/bin" "$HOME/bin" /opt/homebrew/bin /usr/local/bin; do
      case ":$PATH:" in *":$d:"*) ;; *) continue ;; esac
      [ -d "$d" ] && [ -w "$d" ] || continue
      [ ! -e "$d/jitengine" ] && [ ! -L "$d/jitengine" ] || ours "$d/jitengine" || continue   # another jitengine: keep it
      ln -sf "$CMD" "$d/jitengine" && break
    done
  fi
fi
RUN=jitengine; ours "$(command -v jitengine || true)" || RUN=$CMD
say "Done. Serve the local model, or start a coding agent on it:"
printf '\n    %s start       # the OpenAI and Anthropic APIs on http://127.0.0.1:8080\n' "$RUN"
printf '    %s claude      # or codex, opencode, pi\n\n' "$RUN"
if [ "$RUN" = jitengine ]; then :
elif [ "${JITENGINE_NO_MODIFY_PATH:-}" != 1 ]; then echo "New terminals know it as just 'jitengine'; for this one, run: $LINE"
else echo "To call it just 'jitengine', add $HOME_DIR/bin to your PATH."
fi
echo "'jitengine stop' stops the engine."

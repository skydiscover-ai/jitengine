#!/bin/bash
# Install a JITEngine engine and the agent launcher on this Mac, in one command:
#
#   curl -fsSL https://jitengine.ai/install.sh | bash -s -- flint   # or qwen3.6, flash-next, qwen3-1.7b
#
# It picks a build for this Mac's chip and memory (engines/builds.tsv), installs it and
# jit-launch into a Python environment of the build's own in ~/.jitengine, downloads the model at its pinned revision into ~/models (files the Hugging Face cache
# already holds are hard-linked from it, not copied, when it is on the same volume; for Flash-Next, also packs its
# experts for streaming, once), and adds a `jitengine` command: on the PATH of new terminals through the shell
# profile, and of this one through a link in the first of ~/.local/bin, ~/bin, /opt/homebrew/bin and /usr/local/bin
# that is on it and writable. Then `jitengine claude` (or codex, opencode, pi) starts the engine if it is not running,
# waits until it is ready, and starts the agent on it.
#
# Environment: JITENGINE_HOME (default ~/.jitengine), JITENGINE_MODELS (default ~/models), JITENGINE_REPO and
# JITENGINE_REF (the repository and branch to fetch the engines from when this script does not run from a checkout;
# by default the public repository's default branch), JITENGINE_NO_MODIFY_PATH=1 to leave the shell profile and the
# PATH folders alone, and JITENGINE_DRY_RUN=1 to print the build this Mac would get and stop.
set -euo pipefail

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'jitengine install: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: curl -fsSL https://jitengine.ai/install.sh | bash -s -- [flint | qwen3.6 | flash-next | qwen3-1.7b]   (default: flint)

Installs an engine build chosen for this Mac's chip and memory and jit-launch into ~/.jitengine, downloads the model
into ~/models, and adds a `jitengine` command: `jitengine start` serves the model on http://127.0.0.1:8080, and
`jitengine claude` (or codex, opencode, pi) starts the engine if it is not running and the agent on it.

Environment: JITENGINE_HOME, JITENGINE_MODELS, JITENGINE_REPO, JITENGINE_REF, JITENGINE_NO_MODIFY_PATH=1 to leave
the shell profile and the PATH folders alone, and JITENGINE_DRY_RUN=1 to print the build this Mac would get and stop.
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
[ "$MODEL" != flashnext ] || MODEL=flash-next

SELF="${BASH_SOURCE[0]:-}"
if [ -n "$SELF" ] && [ -f "$SELF" ] && [ -f "$(dirname "$SELF")/engines/builds.tsv" ]; then
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

# The build: the first row of engines/builds.tsv for this model whose memory and chip match this Mac (the table says
# why each row is there, from what our two Macs measured). A model with no matching row needs more memory.
ENGINE=""; MIN_G=""; KNOWN=""
while read -r m min_gb chips engine _; do
  case "$m" in ''|'#'*) continue ;; esac
  case " $KNOWN " in *" $m "*) ;; *) KNOWN="$KNOWN $m" ;; esac
  [ "$m" = "$MODEL" ] || continue
  if [ -z "$MIN_G" ] || [ "$min_gb" -lt "$MIN_G" ]; then MIN_G=$min_gb; fi
  if [ -z "$ENGINE" ] && [ "$G" -ge "$min_gb" ]; then
    case ",$chips," in *",*,"*|*",${CHIP#Apple },"*) ENGINE=$engine ;; esac
  fi
done < "$SRC/engines/builds.tsv"
[ -n "$MIN_G" ] || die "unknown model '$MODEL': use one of$KNOWN"
[ -n "$ENGINE" ] || die "the $MODEL engines need a Mac with $MIN_G GB of memory or more (this one has $G GB)"
MAC=${ENGINE##*-}

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
  flash-next:*)
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
  qwen3-1.7b:*)
    REPO_ID=mlx-community/Qwen3-1.7B-4bit; REV=3b1b1768f8f8cf8351c712464f906e86c2b8269e; DIR=Qwen3-1.7B-4bit
    # No memory budget to fit: the engine caps the requests running at once (32,768 padded tokens) and the prompt cache
    # (6 GiB), and MLX peaked at 9.0 to 12.7 GiB in its long-prompt and Claude Code tests.
    NEED_GIB=0; RESIDENT_GIB=0 ;;
  *) die "unknown model '$MODEL': use flint, qwen3.6, flash-next or qwen3-1.7b" ;;
esac
say "This Mac (${CHIP:-Apple silicon}, $G GB) gets the $ENGINE engine"
[ "${JITENGINE_DRY_RUN:-}" != 1 ] || exit 0

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

# A Python environment per build: the builds pin MLX, not all to the same version (Qwen3-1.7B's 0.32.4, the others'
# 0.32.2), and each checks its kernels against its own version's stock operations.
VENV="$HOME_DIR/venvs/$ENGINE"
say "Installing $ENGINE, its runtime and jit-launch into $VENV"
mkdir -p "$HOME_DIR/bin" "$MODELS"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
# The shared packages the build depends on are folders of engines/ (jitengine-flint is engines/flint, jitengine-qwen
# engines/qwen): the build installs with each its pyproject.toml, or its model's engine's, names.
SHARED=("$SRC/runtime")
for LIB in "$SRC"/engines/*/; do
  LIB=${LIB%/}; NAME=${LIB##*/}
  [ -f "$LIB/pyproject.toml" ] && [ "$NAME" != "$ENGINE" ] || continue
  if grep -qs "\"jitengine-$NAME==" "$SRC/engines/$ENGINE/pyproject.toml" "$SRC/engines/${ENGINE%-*}/pyproject.toml"; then
    SHARED+=("$LIB")
  fi
done
"$VENV/bin/pip" install -q --disable-pip-version-check "${SHARED[@]}" "$SRC/engines/$ENGINE[download]" "$SRC/launch"

# A download that stops growing for STALL seconds is stopped and started again, up to five times: files already
# downloaded are kept, and a file cut off midway starts over (huggingface_hub names each attempt's partial file anew,
# so the partial files of stopped attempts are removed once the download completes). On a terminal hf draws its own
# progress bars, of the files and of the bytes; anywhere else (a log, an agent) it draws at most the count of files,
# so then every REPORT seconds this says how much is there.
STALL="${JITENGINE_DOWNLOAD_STALL_SECS:-300}"
REPORT="${JITENGINE_DOWNLOAD_REPORT_SECS:-60}"

# Hard-links the large files the Hugging Face cache already holds of repo $1 at revision $2 into $MODELS/$3
# (jitengine_runtime.hub): the download then checks them in place instead of copying or fetching them, and the folder
# stays where the packs and caches are named from it. A cache on another volume links nothing, and hf copies from it.
# Succeeds when the folder holds files of the cache.
reuse_cache() {
  [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m jitengine_runtime.hub link "$1" "$2" "$MODELS/$3" 2>/dev/null
}

download() {
  say "Downloading $1 into $MODELS/$3 (files already there are kept)"
  local dir="$MODELS/$3" try pid size last idle quiet=0 since=0 cache=""
  [ -t 2 ] || quiet=1
  if reuse_cache "$1" "$2" "$3"; then
    # A linked file that fails hf's check must be fetched afresh: hf would copy the cache's file over its own link.
    # So hf looks in an empty cache for this download.
    cache="$dir/.cache/huggingface/no-hub-cache"; mkdir -p "$cache"
  fi
  for try in 1 2 3 4 5; do
    (
      export HF_HUB_DISABLE_UPDATE_CHECK=1   # no hint about hf's agent skill in the installer's output
      [ -z "$cache" ] || export HF_HUB_CACHE="$cache"
      exec "$VENV/bin/hf" download "$1" --revision "$2" --local-dir "$dir" >/dev/null
    ) &
    pid=$!; last=""; idle=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 5
      size=$(du -sk "$dir" 2>/dev/null | cut -f1 || true)
      if [ "$size" = "$last" ]; then idle=$((idle + 5)); else idle=0; last=$size; fi
      if [ "$quiet" = 1 ]; then
        since=$((since + 5))
        if [ "$since" -ge "$REPORT" ]; then
          since=0; echo "  $(awk -v k="${size:-0}" 'BEGIN {printf "%.1f", k / 1048576}') GiB of $1 is in $dir so far"
        fi
      fi
      if [ "$idle" -ge "$STALL" ]; then
        echo "  the download has not moved for $STALL s; starting it again ($try of 5)"
        kill "$pid" 2>/dev/null || true
        break
      fi
    done
    if wait "$pid"; then
      find "$dir/.cache/huggingface/download" -name "*.incomplete" -type f -delete 2>/dev/null || true
      return 0
    fi
  done
  die "could not download $1 into $dir; run this again to continue"
}
if [ -n "$PACK_VAR" ]; then
  PACK_DIR="$MODELS/flash-next-pack"
  if ! ls "$PACK_DIR/$DIR"-*.experts >/dev/null 2>&1; then
    NEED_GB=$((PACK_GB + 4))
    # the model's 112 GB, unless it is in place or linked from the Hugging Face cache on this volume
    [ -f "$MODELS/$DIR/model.safetensors.index.json" ] || reuse_cache "$REPO_ID" "$REV" "$DIR" || NEED_GB=$((NEED_GB + 112))
    FREE_GB=$(df -k "$MODELS" | awk 'NR == 2 {printf "%d", $4 * 1024 / 1e9}')
    [ "$FREE_GB" -ge "$NEED_GB" ] || die "Flash-Next needs about $NEED_GB GB free on the disk holding $MODELS; it has $FREE_GB GB"
  fi
fi
download "$REPO_ID" "$REV" "$DIR"
[ -z "$EXTRA_REPO" ] || download $EXTRA_REPO
if [ -n "$PACK_VAR" ]; then
  say "Packing the experts for streaming into $PACK_DIR (once, about $PACK_GB GB; a later run only checks it)"
  mkdir -p "$PACK_DIR"
  env "$PACK_VAR=$PACK_DIR" "$VENV/bin/jitengine-$ENGINE" pack --model-path "$MODELS/$DIR" $PACK_ARGS
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

# The command remembers where this installed the engine; JITENGINE_HOME still overrides it.
{ cat <<'EOF'
#!/bin/bash
# jitengine: run a coding agent on the local engine, starting the engine first if it is not running.
#   jitengine claude | codex | opencode | pi [-- agent args]   start the engine if needed, then the agent on it
#   jitengine serve [serve args]                               run the engine in this terminal
#   jitengine start | restart | stop | status | logs
#   jitengine bench [--report]                                 time the engine; --report also sends the numbers
# If the engine's default memory budget does not fit in the memory available now, start passes the largest budget
# that does (the engine then reads more from the SSD and runs slower); JITENGINE_BUDGET_GIB=N sets one yourself.
set -euo pipefail
EOF
printf 'INSTALLED_HOME=%q\n' "$HOME_DIR"
cat <<'EOF'
HOME_DIR="${JITENGINE_HOME:-$INSTALLED_HOME}"
. "$HOME_DIR/engine.env"
BIN="$HOME_DIR/venvs/${ENGINE_CMD#jitengine-}/bin"   # the build's own Python environment
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
  [ "$NEED_GIB" != 0 ] || return 0   # an engine with no memory budget (Qwen3-1.7B caps its caches itself)
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
  printf '%s\n' "$SERVE_ARGS $budget" > "$HOME_DIR/serve.args"   # what jitengine bench reports
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
  ""|-h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) start; JITENGINE_HOME="$HOME_DIR" exec "$BIN/jit-launch" "$@" ;;
esac
EOF
} > "$HOME_DIR/bin/jitengine"
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

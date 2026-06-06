#!/bin/bash
# sync-zonkey.sh - Push the Zonkey *codebase* (Python + config JSON) to the
# Windows GPU box via scp. Code only: no data, no checkpoints, no caches.
#
#   Files sent: every *.py (dir structure preserved) + configs/*.json
#   Not sent:   HF dataset cache, checkpoints, logs, .git, __pycache__, this script
#
# Usage:
#   ./sync-zonkey.sh            # transfer the codebase
#   ./sync-zonkey.sh --dry-run  # list what would transfer, change nothing

set -u

# ----------------------------- Configuration ------------------------------
REMOTE_HOST="Owner@192.168.1.178"            # user@host of the GPU box
REMOTE_BASE="C:/Users/Owner/projects/Zonkey" # remote dest (forward slashes; cmd.exe shell)
SSH_KEY="~/.ssh/id_ed25519"
# LOCAL_PATH is auto-detected from this script's location (robust across machines).
LOCAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ---------------------------------------------------------------------------

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

SSH_KEY="${SSH_KEY/#\~/$HOME}"
SSH_OPTS=(-i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

echo -e "${BLUE}🔍 Zonkey codebase sync (scp)${NC}"
echo "=================================================="
echo -e "${BLUE}📁 Local: ${NC}$LOCAL_PATH"
echo -e "${BLUE}🖥️  Remote:${NC} $REMOTE_HOST:$REMOTE_BASE"
[ "$DRY" -eq 1 ] && echo -e "${YELLOW}(dry run — nothing will be written)${NC}"
echo ""

cd "$LOCAL_PATH" || { echo -e "${RED}❌ Cannot cd to $LOCAL_PATH${NC}"; exit 1; }

# Build the file list: all .py (minus .git/__pycache__/local test) + configs/*.json
FILES=()
while IFS= read -r f; do FILES+=("${f#./}"); done < <(
    find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' -not -name '_fm_smoke_test.py'
    find ./configs -maxdepth 1 -name '*.json'
)
if [ "${#FILES[@]}" -eq 0 ]; then
    echo -e "${RED}❌ No files matched${NC}"; exit 1
fi
echo -e "${GREEN}📊 ${#FILES[@]} files to transfer${NC}"

if [ "$DRY" -eq 1 ]; then
    printf '  %s\n' "${FILES[@]}"
    exit 0
fi

# Test SSH
echo -e "${BLUE}🔐 Testing SSH...${NC}"
if ! ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "echo ok" >/dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to $REMOTE_HOST${NC}"; exit 1
fi
echo -e "${GREEN}✅ Connected${NC}"

# Create the remote directory tree in one shot (cmd.exe; backslashes; ignore "exists").
echo -e "${BLUE}📁 Creating remote directories...${NC}"
DIRS=$(for f in "${FILES[@]}"; do d=$(dirname "$f"); [ "$d" = "." ] && echo "" || echo "$d"; done | sort -u)
MKCMD="mkdir \"${REMOTE_BASE//\//\\}\" 2>nul"
while IFS= read -r d; do
    [ -z "$d" ] && continue
    win="${REMOTE_BASE}/${d}"; win="${win//\//\\}"
    MKCMD="$MKCMD & mkdir \"$win\" 2>nul"
done <<< "$DIRS"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$MKCMD" >/dev/null 2>&1

# Transfer
echo -e "${BLUE}🚀 Copying...${NC}"
ok=0; fail=0; n=0; total=${#FILES[@]}
for f in "${FILES[@]}"; do
    n=$((n+1))
    if scp "${SSH_OPTS[@]}" "$f" "$REMOTE_HOST:$REMOTE_BASE/$f" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} [$n/$total] $f"; ok=$((ok+1))
    else
        echo -e "  ${RED}✗${NC} [$n/$total] $f"; fail=$((fail+1))
    fi
done

echo ""
echo -e "${BLUE}📊 ${GREEN}$ok ok${NC}, ${RED}$fail failed${NC} of $total"
[ "$fail" -eq 0 ] && echo -e "${GREEN}✨ Codebase synced!${NC}" || { echo -e "${YELLOW}⚠️  Partial sync${NC}"; exit 1; }

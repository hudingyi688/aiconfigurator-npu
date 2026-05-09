#!/usr/bin/env bash
# Apply aiconfigurator-npu patches to upstream aiconfigurator repo.
#
# Usage:
#   ./tools/apply_patches.sh /path/to/aiconfigurator
#
# Run this after cloning or updating the upstream aiconfigurator repo.

set -euo pipefail

UPSTREAM="${1:-}"
if [[ -z "$UPSTREAM" ]]; then
    echo "Usage: $0 /path/to/aiconfigurator" >&2
    exit 1
fi

if [[ ! -d "$UPSTREAM/.git" ]]; then
    echo "Error: $UPSTREAM is not a git repository" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"

echo "Applying patches to: $UPSTREAM"

for patch in "$PATCHES_DIR"/*.patch; do
    echo "  Applying: $(basename "$patch")"
    git -C "$UPSTREAM" apply --check "$patch" 2>/dev/null \
        && git -C "$UPSTREAM" apply "$patch" \
        || echo "  [SKIP] Already applied or conflict: $(basename "$patch")"
done

echo ""
echo "Copying system data..."
DATA_SRC="$SCRIPT_DIR/../systems/data/ascend_910b"
DATA_DST="$UPSTREAM/src/aiconfigurator/systems/data/ascend_910b"

if [[ -d "$DATA_SRC" ]]; then
    mkdir -p "$DATA_DST"
    cp -r "$DATA_SRC/." "$DATA_DST/"
    echo "  Copied: $DATA_SRC → $DATA_DST"
fi

YAML_SRC="$SCRIPT_DIR/../systems/ascend_910b_aiconfigurator/ascend_910b.yaml"
YAML_DST="$UPSTREAM/src/aiconfigurator/systems/ascend_910b.yaml"
if [[ -f "$YAML_SRC" ]]; then
    cp "$YAML_SRC" "$YAML_DST"
    echo "  Copied: ascend_910b.yaml"
fi

echo ""
echo "Done. Run 'pip install -e $UPSTREAM' to reinstall."

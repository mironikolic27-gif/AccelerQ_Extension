#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <backup_name>"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_NAME="$1"
DEST="$REPO_ROOT/backup_runs/$BACKUP_NAME"

if [ -e "$DEST" ]; then
  echo "Error: destination already exists: $DEST"
  exit 1
fi

move_if_exists() {
  local src="$1"
  local dest_dir="$2"
  if [ -f "$src" ]; then
    mv "$src" "$dest_dir/"
    echo "moved: $src -> $dest_dir/"
  else
    echo "warning: missing, skipped: $src"
  fi
}

mkdir -p "$DEST/src"

for f in logger.txt; do
  move_if_exists "$REPO_ROOT/src/$f" "$DEST/src"
done

for impl_dir in "$REPO_ROOT/data"/*/; do
  impl="$(basename "$impl_dir")"
  # Skip objective-scoped analysis output dirs (backed up separately below).
  case "$impl" in
    O1|O2|O3|O4) continue ;;
  esac
  dest_impl="$DEST/data/$impl"
  mkdir -p "$dest_impl"

  for f in \
    "records_${impl}.csv" \
    "hamiltonian_${impl}.csv" \
    "energies_stage3_${impl}.csv" \
    "stage2_profiling_${impl}.csv" \
    "stage3_memory_series_${impl}.csv" \
    logger.txt; do
    move_if_exists "$impl_dir/$f" "$dest_impl"
  done

  if [ "$impl" = "NN_HEA" ]; then
    move_if_exists "$impl_dir/optimal_NN_HEA.csv" "$dest_impl"
    move_if_exists "$impl_dir/stage2b_profiling_NN_HEA.csv" "$dest_impl"
  fi
done

shopt -s nullglob
for odir in O1 O2 O3 O4; do
  src_dir="$REPO_ROOT/data/$odir"
  dest_dir="$DEST/data/$odir"
  mkdir -p "$dest_dir"
  if [ -d "$src_dir" ]; then
    for f in "$src_dir"/*; do
      [ -f "$f" ] || continue
      move_if_exists "$f" "$dest_dir"
    done
  fi
done
shopt -u nullglob

echo "Backup complete: $DEST"

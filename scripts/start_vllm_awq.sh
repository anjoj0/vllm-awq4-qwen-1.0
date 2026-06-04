#!/usr/bin/env bash
set -euo pipefail

AWQ_MMQ_DIR="${AWQ_MMQ_DIR:-/root/csrc/awq_mmq_gfx1151}"
AWQ_MMQ_PKG="$AWQ_MMQ_DIR/awq_mmq_gfx1151"

if [ -d "$AWQ_MMQ_PKG" ] && [ -f "$AWQ_MMQ_DIR/setup.py" ]; then
  AWQ_MMQ_SO="$(find "$AWQ_MMQ_PKG" -maxdepth 1 -name "_C*.so" -print -quit)"
  AWQ_MMQ_NEED_BUILD=0
  if [ -z "$AWQ_MMQ_SO" ]; then
    AWQ_MMQ_NEED_BUILD=1
  elif find "$AWQ_MMQ_DIR" -maxdepth 2 \( -name "*.cpp" -o -name "*.hip" -o -name "*.py" \) -newer "$AWQ_MMQ_SO" -print -quit | grep -q .; then
    AWQ_MMQ_NEED_BUILD=1
  fi

  if [ "$AWQ_MMQ_NEED_BUILD" = "1" ]; then
    echo "Building AWQ MMQ custom op in $AWQ_MMQ_DIR"
    (cd "$AWQ_MMQ_DIR" && python setup.py build_ext --inplace --build-temp "$AWQ_MMQ_DIR/build/temp.linux-x86_64-cpython-312")
  fi
fi

exec "$@"

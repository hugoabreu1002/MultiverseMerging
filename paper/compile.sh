#!/usr/bin/env bash
set -euo pipefail

# Simple LaTeX compile helper for the paper directory.
# Usage: ./compile.sh [file.tex]

TEXFILE="${1:-merging_universes_geometry_comparison.tex}"
if [ ! -f "$TEXFILE" ]; then
  echo "TeX file not found: $TEXFILE" >&2
  exit 1
fi

echo "Compiling $TEXFILE"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -file-line-error "$TEXFILE"
  echo "Done: ${TEXFILE%.*}.pdf"
  exit 0
fi

# Fallback: run pdfLaTeX / BibTeX sequence
pdflatex -interaction=nonstopmode -file-line-error "$TEXFILE"

# if .aux references a bibliography, run bibtex
if grep -q "\\bibliography{\|\\addbibresource" <<< "$(cat ${TEXFILE})" 2>/dev/null || [ -f "${TEXFILE%.*}.aux" ]; then
  bibtex "${TEXFILE%.*}" 2>/dev/null || true
fi

pdflatex -interaction=nonstopmode -file-line-error "$TEXFILE"
pdflatex -interaction=nonstopmode -file-line-error "$TEXFILE"

echo "Done: ${TEXFILE%.*}.pdf"

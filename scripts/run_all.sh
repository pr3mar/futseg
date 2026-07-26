#!/usr/bin/env bash
# Run futseg over every photo in input/, reproducibly.
#
# This drives the CLI rather than reimplementing it: no second argument parser,
# no second set of defaults to drift out of sync. Everything is overridable by
# environment variable so a different prompt or model is a one-liner.
#
#     scripts/run_all.sh
#     PROMPT="a quiet forest" MODEL=sdxl-inpaint scripts/run_all.sh
#     BACKEND=composite scripts/run_all.sh          # no diffusion weights needed
#
# From the host: make exec CMD="scripts/run_all.sh"
#
# Exits non-zero if any photo failed, so it is usable in a pipeline. A photo with
# no person in it is reported and counted as a failure by the CLI's own exit 1.

set -uo pipefail

INPUT="${INPUT:-input}"
OUT="${OUT:-out}"
QUALITY="${QUALITY:-best}"
BACKEND="${BACKEND:-diffusion}"
MODEL="${MODEL:-flux2-klein}"
PROMPT="${PROMPT:-Cinematic photo, ergonomic spaceship bridge. Three crew members in \
matte-black tactical helmets and charcoal flight armor, hands gripping dual joystick \
controls. Massive multi-paned viewscreen shows an overwhelming, sharp vortex of white and \
blue light filaments. Multiple glowing multi-layered purple holographic interfaces on the \
console. Interior reflections, intense atmosphere.}"

command -v futseg >/dev/null || {
  echo "futseg is not on PATH -- run this inside the container (make shell)" >&2
  exit 2
}

shopt -s nullglob nocaseglob
photos=("$INPUT"/*.jpg "$INPUT"/*.jpeg "$INPUT"/*.png)
shopt -u nocaseglob

if [ ${#photos[@]} -eq 0 ]; then
  echo "no photos in $INPUT/" >&2
  exit 2
fi

echo "photos:  ${#photos[@]} in $INPUT/"
echo "backend: $BACKEND${BACKEND:+ }${MODEL}"
echo "quality: $QUALITY"
echo "prompt:  ${PROMPT:0:72}..."
echo

failed=0
for photo in "${photos[@]}"; do
  stem=$(basename "${photo%.*}")
  if futseg run "$photo" \
      --prompt "$PROMPT" \
      --out "$OUT/$stem-final.png" \
      --backend "$BACKEND" \
      --model "$MODEL" \
      --quality "$QUALITY"; then
    :
  else
    echo "  ^ $stem failed (exit $?)" >&2
    failed=$((failed + 1))
  fi
done

echo
if [ "$failed" -gt 0 ]; then
  echo "$failed of ${#photos[@]} failed" >&2
  exit 1
fi
echo "all ${#photos[@]} written to $OUT/"

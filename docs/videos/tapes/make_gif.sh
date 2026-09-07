#!/usr/bin/env bash
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
# Converts the MP4s produced by the readme_try_it_out tapes into GIFs. GitHub
# does not play an MP4 committed at a relative path, so the README needs a GIF;
# the MP4s are kept for the documentation site.
#
# VHS records at 2540x1380 (2x a 100-column terminal, Padding 0). Scaling that
# master all the way to 760px is what made the GIF look soft next to the MP4.
# An integer half-scale (1270px) at 15 fps keeps the glyphs close to the MP4
# (a 760px GIF looked soft next to it). The large-file hook allows 3000 KB
# for this. GitHub then downscales to the README column. A 20px
# matching-colour extend-pad is the 12px gutter at 760px, scaled up with
# the frame.
#
# Usage: ./docs/videos/tapes/make_gif.sh [name ...]
# With no arguments both the default and black background variants are built.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VIDEO_DIR="${REPO_ROOT}/docs/videos"
# Half of the 2540px VHS master. Exact 2x so lanczos is not resampling off-grid.
GIF_WIDTH=1270
PAD=20

pad_color_for() {
  case "$1" in
    *_black) echo "0x000000" ;;
    *) echo "0x1E1E2E" ;; # Catppuccin Mocha base
  esac
}

names=("$@")
if [[ ${#names[@]} -eq 0 ]]; then
  names=(readme_try_it_out readme_try_it_out_black)
fi

for name in "${names[@]}"; do
  src="${VIDEO_DIR}/${name}.mp4"
  dst="${VIDEO_DIR}/${name}.gif"

  if [[ ! -f "${src}" ]]; then
    echo "Skipping ${name}: ${src} does not exist" >&2
    continue
  fi

  pad_color="$(pad_color_for "${name}")"
  pad_w=$((GIF_WIDTH + PAD * 2))

  ffmpeg -y -loglevel error -i "${src}" -filter_complex "\
[0:v]fps=15,scale=${GIF_WIDTH}:-2:flags=lanczos,\
pad=${pad_w}:ih+${PAD}*2:${PAD}:${PAD}:${pad_color},split[a][b];\
[a]palettegen=max_colors=256:stats_mode=full[p];\
[b][p]paletteuse=dither=none:diff_mode=rectangle" \
    "${dst}"

  ls -lh "${dst}"
done

#!/usr/bin/env bash
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
# Converts the MP4s produced by the readme_try_it_out tapes into GIFs. GitHub
# does not play an MP4 committed at a relative path, so the README needs a GIF;
# the MP4s are kept for the documentation site.
#
# VHS records at 2540x1380 (2x a 100-column terminal, no padding — padding is
# applied by scaling inside VHS and was the main source of blur). The GIF is
# then scaled to 760px, about the GitHub README column, so each displayed
# pixel averages several source pixels and the glyphs look crisp rather than
# showing every anti-aliased halo.
#
# Usage: ./docs/videos/tapes/make_gif.sh [name ...]
# With no arguments both the default and black background variants are built.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VIDEO_DIR="${REPO_ROOT}/docs/videos"
# Matches the GitHub README column closely enough that the GIF is not scaled
# up in the page, which is what made the 1376px version look soft.
GIF_WIDTH=760

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

  ffmpeg -y -loglevel error -i "${src}" -filter_complex "\
[0:v]fps=15,scale=${GIF_WIDTH}:-2:flags=lanczos,unsharp=5:5:0.8:5:5:0.0,split[a][b];\
[a]palettegen=max_colors=256:stats_mode=full[p];\
[b][p]paletteuse=dither=none:diff_mode=rectangle" \
    "${dst}"

  ls -lh "${dst}"
done

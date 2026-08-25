# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
# Records an extended version of the README "Try It Out" walkthrough on the
# Catppuccin Mocha background. readme_try_it_out_black.tape is the same
# recording on pure black; the steps live in readme_try_it_out_steps.tape.
#
# Run setup_demo_env.sh first: it builds the isolated venv, demo directory,
# ado context, vimrc and Ray head node that the steps assume at
# /tmp/ado-readme-demo. Record from the repository root so the Output and
# Source paths resolve:
#
#   ./docs/videos/tapes/setup_demo_env.sh
#   vhs docs/videos/tapes/readme_try_it_out.tape
#   ./docs/videos/tapes/make_gif.sh
#
# Geometry: FontSize/Width/Height below give a 100x29 terminal. 100 columns is
# the narrowest width at which `ado show measurements` still prints mass,
# volume and density without truncating them; 29 rows is the shortest that
# holds `ado describe experiment` without scrolling.
#
# Values are 2x a 100x29 layout and Padding is 0. VHS applies padding by
# scaling the frame, which is what made the glyphs look soft; recording at
# 2x and letting make_gif.sh scale down is the workaround recommended in
# charmbracelet/vhs#69 and #625.
#
# PlaybackSpeed 2 renders the whole recording at double speed, which halves the
# ~40s that ado spends on Ray startup, the random walk and Ray teardown. Sleep
# and TypingSpeed values are therefore written at twice their intended
# on-screen duration.

Output "docs/videos/readme_try_it_out.mp4"

Set Shell "bash"
Set FontSize 40
Set Padding 0
Set Width 2540
Set Height 1380
Set Framerate 30
Set PlaybackSpeed 2
Set TypingSpeed 60ms
Set CursorBlink false
Set Theme "Catppuccin Mocha"

# HOME drives typer.get_app_dir("ado"), so this points ado at the throwaway
# context and SQLite database created by setup_demo_env.sh.
Env HOME "/tmp/ado-readme-demo/home"
# Stop Ray forwarding worker stdout to the driver; it buries the CLI output.
Env RAY_LOG_TO_DRIVER "0"
# VHS defaults to NO_COLOR=1 and FORCE_COLOR=0, which makes rich render ado's
# output without any colour. Clearing them restores the normal CLI palette.
Env NO_COLOR ""
Env FORCE_COLOR "1"
Env PS1 "$ "

Source "docs/videos/tapes/readme_try_it_out_steps.tape"

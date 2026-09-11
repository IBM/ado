# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
# Pure black background variant of readme_try_it_out.tape. Everything except
# the theme is identical; see that file for the geometry and timing notes and
# readme_try_it_out_steps.tape for the command sequence.
#
#   ./docs/videos/tapes/setup_demo_env.sh
#   vhs docs/videos/tapes/readme_try_it_out_black.tape
#   ./docs/videos/tapes/make_gif.sh
#
# The palette is Catppuccin Mocha with the background swapped for #000000, so
# the two variants differ only in how the terminal surface reads against the
# page rather than in the colour of ado's output.

Output "docs/videos/readme_try_it_out_black.mp4"

Set Shell "bash"
Set FontSize 40
Set Padding 0
Set Width 2540
Set Height 1380
Set Framerate 30
Set PlaybackSpeed 2
Set TypingSpeed 60ms
Set CursorBlink false
Set Theme { "name": "Catppuccin Mocha Black", "background": "#000000", "foreground": "#CDD6F4", "cursor": "#F5E0DC", "selection": "#585B70", "black": "#45475A", "red": "#F38BA8", "green": "#A6E3A1", "yellow": "#F9E2AF", "blue": "#89B4FA", "magenta": "#F5C2E7", "cyan": "#94E2D5", "white": "#BAC2DE", "brightBlack": "#585B70", "brightRed": "#F38BA8", "brightGreen": "#A6E3A1", "brightYellow": "#F9E2AF", "brightBlue": "#89B4FA", "brightMagenta": "#F5C2E7", "brightCyan": "#94E2D5", "brightWhite": "#A6ADC8" }

Env HOME "/tmp/ado-readme-demo/home"
Env RAY_LOG_TO_DRIVER "0"
Env NO_COLOR ""
Env FORCE_COLOR "1"
Env PS1 "$ "

Source "docs/videos/tapes/readme_try_it_out_steps.tape"

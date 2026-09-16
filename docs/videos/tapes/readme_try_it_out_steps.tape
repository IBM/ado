# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
# Shared command sequence for the README "Try It Out" recording. This file has
# no Output or Set directives and cannot be run on its own; it is pulled in by
# readme_try_it_out.tape and readme_try_it_out_black.tape, which differ only in
# the terminal theme.
#
# Each step is captioned with a typed bash comment. Interactive bash treats a
# leading # as a no-op, so the caption appears in the transcript without
# running anything.

Hide
Type@0 "export PATH=/tmp/ado-readme-demo/venv/bin:$PATH" Enter
Type@0 "cd /tmp/ado-readme-demo/demo" Enter
Type@0 "clear" Enter
Wait+Line /^\$$/
Sleep 1s
Show

Require ado

Type "# List the installed experiments" Enter
Type "ado get experiments" Sleep 1s Enter
Wait+Line@60s /^\$$/
Sleep 4s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

Type "# What does the calculate_density experiment need, and what does it measure?" Enter
Type "ado describe experiment calculate_density" Sleep 1s Enter
Wait+Line@60s /^\$$/
Sleep 5s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

# vim rather than cat, so the decorator and signature are syntax highlighted.
# vim draws on the alternate screen and never returns a prompt, so the vim
# steps wait on a fixed Sleep instead of Wait+Line before quitting.
Type "# The experiment is just a decorated Python function" Enter
Type "vim -R density/density.py" Sleep 1s Enter
Sleep 8s
Type ":q" Enter
Wait+Line@60s /^\$$/
Sleep 1s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

Type "# Define the space to explore" Enter
Type "vim -R space.yaml" Sleep 1s Enter
Sleep 8s
Type ":q" Enter
Wait+Line@60s /^\$$/
Sleep 1s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

Type "# Create it" Enter
Type "ado create space -f space.yaml" Sleep 1s Enter
Wait+Line@60s /^\$$/
Sleep 3s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

Type "# Explore it" Enter
Type "ado create operation -f operation.yaml --use-latest discoveryspace" Sleep 1s Enter
Wait+Line@180s /^\$$/
Sleep 4s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

# --property keeps the table inside 100 columns. Naming mass and volume as well
# as density is what makes them readable: with density alone rich spends the
# width on the 36 character experiment_id column and truncates every value.
Type "# View the results" Enter
Type "ado show measurements operation --use-latest --property mass --property volume --property density" Sleep 1s Enter
Wait+Line@60s /^\$$/
Sleep 6s

Hide
Type@0 "clear" Enter
Wait+Line /^\$$/
Show

Type "# Every space and operation is recorded" Enter
Type "ado get spaces" Sleep 1s Enter
Wait+Line@60s /^\$$/
Sleep 2s
Type "ado get operations" Sleep 1s Enter
Wait+Line@60s /^\$$/
Sleep 5s

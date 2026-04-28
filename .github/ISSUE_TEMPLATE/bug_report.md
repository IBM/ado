---
name: Bug report
about: Create a report to help us improve
title: "bug: "
labels: bug
assignees: ""
---

## Issue Description

A clear and concise description of what the bug is.

### How to reproduce

Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected behaviour

A clear and concise description of what you expected to happen.

### Screenshots/Logs

If applicable, add screenshots and logs (e.g., Ray Logs) to help explain your
problem.

### Python/ado/system info

Please include the output of:

```terminaloutput
python --version
ado version
Your OS
```

**Note:** If you installed ado in editable mode (e.g., `pip install -e .`) or
ran `uv sync`, the version metadata may not be up to date. Please reinstall to
get an accurate version number:

```bash
# If you used uv sync:
uv sync --reinstall

# If you used pip install -e:
pip install -e . --force-reinstall --no-deps
```

## Additional information

Add any other context about the problem here.

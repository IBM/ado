# review

- Review the committed changes in the current branch
- Be picky
- Use [AGENTS.md](../../AGENTS.md)
and [plugin-development.md](../rules/plugin-development.mdc) to evaluate code
- Link comments to code
- Format report with following sections (omit if not relevant)
  - Overview: What is changed
  - Bugs: Any potential bugs
  - Critical Issues: Critical problems with implementation or
    structure
  - Medium Issues: Issues that should be considered but may not need to be
    fixed now
  - Style Issues: Any style changes that should be made
  - Linting Issues: Issues raised by the linters. black, ruff or
    markdownlint-cli2 lints will block commits
- Write review to md file also

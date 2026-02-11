schema_version = 1

project {
  license        = "MIT"
  copyright_year = 2025
  copyright_holder = "IBM Corporation"

  # (OPTIONAL) A list of globs that should not have copyright/license headers.
  # Supports doublestar glob patterns for more flexibility in defining which
  # files or folders should be ignored
  header_ignore = [
    ".cra/**",
    ".eggs/**",
    ".git/**",
    ".github/**",
    ".idea/**",
    ".pre-commit-config.yaml",
    ".tox/**",
    ".venv/**",
    ".vscode/**",
    "**build/lib/**",
    "dist/**",
    "toxenv/**",
  ]
}

schema_version = 1

project {
  license        = "MIT"
  copyright_year = 2025
  copyright_holder = "IBM Corporation"

  # Globs for paths that should not have copyright/license headers.
  # A directory and the files inside it are different paths, so each directory
  # is listed twice. A leading **/ makes the pattern match at any depth.
  header_ignore = [
    # Directories
    "**/.cra",
    "**/.eggs",
    "**/.git",
    "**/.github",
    "**/.idea",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.tox",
    "**/.venv",
    "**/.vscode",
    "**/__pycache__",
    "**/build",
    "**/dist",
    "**/node_modules",
    "**/toxenv",

    # Contents of those directories
    "**/.cra/**",
    "**/.eggs/**",
    "**/.git/**",
    "**/.github/**",
    "**/.idea/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/.tox/**",
    "**/.venv/**",
    "**/.vscode/**",
    "**/__pycache__/**",
    "**build/lib/**",
    "**/build/**",
    "**/dist/**",
    "**/node_modules/**",
    "**/toxenv/**",

    # Individual files and non-source data
    ".pre-commit-config.yaml",
    "**/*.lp",
    "**/*.log",
    "**/*.mp4",
    "**/*.mps",
    "**/*.mps.gz",
    "**/*.mst",
  ]
}

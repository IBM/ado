# Writing Agent Skills

Agent skills in `.agents/skills/` must conform to the
[Agent Skills Specification](https://agentskills.io/home) (see also
[upstream documentation](https://github.com/agentskills/agentskills/tree/main/docs/skill-creation)):

- **Directory Layout**:
  - `SKILL.md` (required): metadata frontmatter and concise instructions
    (< 500 lines / ~5,000 tokens).
  - `references/` (optional): on-demand reference documentation, secondary
    guides, and deep schemas.
  - `assets/` (optional): static resources, configuration templates, and YAML
    example manifests.
  - `scripts/` (optional): executable helper scripts.
- **Frontmatter**:
  - `name` (required): 1–64 characters, lowercase alphanumeric and hyphens
    only, matching the parent directory name exactly.
  - `description` (required): non-empty, maximum 1024 characters. Must use
    imperative phrasing ("Use when...") specifying clear activation
    conditions, intent, and domain context.
- **Progressive Disclosure**:
  - Keep `SKILL.md` focused on core workflows, decision rules, and high-value
    gotchas.
  - Move extended reference guides into `references/` and link them with
    explicit loading triggers.
  - Store reusable YAML manifests, templates, and example files in `assets/`
    and link them using relative paths (e.g. `assets/my-example.yaml`).
  - Write automated tests under `tests/` for all YAML examples and templates
    stored in `assets/`.
- **Quality & Scope**:
  - Be brief and to the point; avoid ambiguous statements.
  - Avoid duplication — check existing skills under `.agents/skills/`,
    examples under `examples/`, and documentation under `docs/`.
  - After writing or editing a skill, verify that all referenced file and
    directory paths exist.

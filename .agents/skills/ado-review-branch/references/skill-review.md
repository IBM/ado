# Skill Review

Agent skills are text documents under `.agents/skills/`. Apply this procedure
to every new or modified skill on the branch.

## Procedure

1. Fetch the upstream Agent Skills documentation (see below).
2. Verify specification conformance.
3. Read every file in the skill directory, including each file listed in the
   skill's References section.
4. Compare the skill against related content in `.agents/skills/`,
   `examples/`, `docs/`, the root [AGENTS.md](../../../../AGENTS.md), and the
   skill-writing rules in [.agents/skills/AGENTS.md](../../AGENTS.md). Compare
   the `description` of all other skills and identify any whose scope overlaps.
5. Write one subsection per skill using the structure below.

## Upstream documentation

[.agents/skills/AGENTS.md](../../AGENTS.md) links to the upstream
specification and skill-creation documentation. Fetch the current content of
those links, and of the pages they reference on skill content and descriptions,
before reviewing. Review against what was fetched instead of relying on memory.

If the documentation cannot be fetched, state that in the review and fall back
to [.agents/skills/AGENTS.md](../../AGENTS.md).

## Specification conformance

Check the skill against the fetched documentation and
[.agents/skills/AGENTS.md](../../AGENTS.md). In addition, check that:

- `name` matches the parent directory, uses only lowercase alphanumerics and
  hyphens, is at most 64 characters, and does not start, end, or contain
  consecutive hyphens (`--`)
- `description` is at most 1024 characters, states what the skill does and when
  to use it, and uses imperative phrasing ("Use when...")
- `SKILL.md` is under 500 lines
- supplementary docs are in `references/`, templates and example manifests in
  `assets/`, helper scripts in `scripts/`
- file references from `SKILL.md` are one level deep (no reference chains)
- every YAML file in `assets/` is covered by a test under `tests/`

## Per-skill structure

Keep each part short; omit a part that has nothing to report.

1. **What the skill adds** — one or two sentence summary
2. **Related existing documentation** — bullet list of related skills,
   examples, and website docs, with a note on why each is related
3. **New information** — bullet list of genuinely new content not already in
   other skills, AGENTS.md, ado-develop-plugin, or linked docs. Base this on
   the files and descriptions read in the procedure above.
4. **Critical Issues** — subdivide into "Duplications" and "Misplacements"
   - Duplications: for each addition that duplicates existing content
     - name the section
     - state what it duplicates
     - give the exact link or section to replace it with, or state "remove —
       covered via general link to X"
   - Misplacements: for each piece of new information that is misplaced
     - content outside the scope of the skill's `description` field
     - content that does not match the objective of the skill
     - state where it should be moved to (another skill, AGENTS.md,
       ado-develop-plugin, etc.)
5. **Medium Issues**
   - Inline YAML or code that should be moved to `assets/` (or replaced with a
     link to an existing example file)
   - Assets in `assets/` without a test
   - Files in `references/` or `assets/` not linked from `SKILL.md` with a
     clear loading trigger
   - Content the agent already knows, or generic advice without
     project-specific detail
   - Menus of equal options where a default should be given
   - Formatting problems, structural issues, or misleading text
   - Mistakes in code examples
   - Inconsistencies with other skill files or agent files
   - File or directory paths referenced in the skill that do not exist in the
     repo
6. **Recommendation** — one line stating whether the skill should be (a) kept
   largely as is, (b) changed substantially, or (c) removed. If substantial
   changes are required, add a short sketch of the revised skill.

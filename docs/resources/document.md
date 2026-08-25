<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
A `document` resource stores a markdown or HTML report or note in the
metastore. Use it to persist and share analysis write-ups about operations,
spaces, or a whole project.

## Creating a `document`

Create a document from a YAML configuration:

```shell
ado create document -f document.yaml
```

Example markdown configuration (`contentType` defaults to `markdown` if
omitted):

```yaml
metadata:
  name: Example operation report
  description: Summary report for an operation
contentType: markdown
content: |
  # Operation report

  This is an example document resource.
relatedResources:
  - id: operation-test-12345678
    role: parent
```

Example HTML configuration:

```yaml
metadata:
  name: Example HTML report
  description: Summary report stored as HTML
contentType: html
content: |
  <html><body><h1>Operation report</h1><p>Example HTML body.</p></body></html>
relatedResources:
  - id: operation-test-12345678
    role: parent
```

- `content` — body of the document (required).
- `contentType` — `markdown` (default) or `html`.
- `relatedResources` — optional list of related ado resources. Each entry has:
  - `id` — resource identifier
  - `role`
    - `parent`: the report is about this resource
    - `child`: the document outlines the motivation for creating the resource
- `metadata` — optional name, description, and labels.

Validate without persisting:

```shell
ado create document -f document.yaml --dry-run
```

Generate a starter template:

```shell
ado template document
```

## Reading a `document`

List or fetch documents:

```shell
ado get document
ado get document document-abc12345 -o yaml
ado get document document-abc12345 -o config
```

Filter by related resource or metadata name (paths are under `config.`):

```shell
ado get document -q 'config.relatedResources.id=operation-test-12345678'
ado get document -q 'config.metadata.name=Example operation report'
```

Describe a document:

```shell
ado describe document document-abc12345
ado describe document document-abc12345 > report.md
```

Markdown is rendered with rich; HTML is printed as the HTML source. Rich
handles terminal vs redirected output.

## Deleting a `document`

```shell
ado delete document document-abc12345
```

## Editing metadata

```shell
ado edit document document-abc12345
```

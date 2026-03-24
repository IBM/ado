# Legacy Validators

Legacy validators help you upgrade old resource files that use deprecated fields
or formats. This guide covers the legacy validator system and its advanced
features.

## Overview

The legacy validator system provides:

- **Automatic dependency resolution**: Validators can depend on other validators
- **Smart error messages**: Get helpful suggestions when validation fails
- **Progress tracking**: Visual feedback during upgrade operations
- **Automatic ordering**: Validators run in the correct order automatically
- **Dry-run mode**: Preview changes before applying them

## Quick Start

### List Available Validators

View all legacy validators:

```bash
ado legacy list
```

View validators for a specific resource type:

```bash
ado legacy list samplestore
```

### Get Validator Information

Get detailed information about a specific validator:

```bash
ado legacy info discoveryspace_properties_field_removal
```

This shows:

- Description of what the validator does
- Which deprecated fields it handles
- Version information
- Usage examples

### Upgrade Resources

#### Upgrade Resources in Metastore

Apply a legacy validator to resources in your metastore:

```bash
ado upgrade discoveryspace --apply-legacy-validator discoveryspace_properties_field_removal
```

Apply multiple validators (they will be automatically ordered):

```bash
ado upgrade samplestore \
  --apply-legacy-validator samplestore_kind_entitysource_to_samplestore \
  --apply-legacy-validator samplestore_module_type_entitysource_to_samplestore
```

#### Upgrade Local YAML Files

Upgrade local YAML files without loading them into the metastore:

```bash
ado legacy upgrade --file my-resource.yaml \
  --apply-legacy-validator discoveryspace_properties_field_removal
```

Upgrade multiple files:

```bash
ado legacy upgrade \
  --file resource1.yaml \
  --file resource2.yaml \
  --apply-legacy-validator validator_name
```

## Advanced Features

### Automatic Dependency Resolution

Validators can depend on other validators. When you specify a validator, the
system automatically:

1. Includes all required dependencies
2. Orders validators correctly using topological sort
3. Notifies you when dependencies are auto-included

**Example:**

If validator B depends on validator A, you only need to specify B:

```bash
ado upgrade samplestore --apply-legacy-validator validator_b
```

The system will automatically:

- Include validator A
- Run A before B
- Show you: "Auto-included dependencies: validator_a"

### Circular Dependency Detection

The system detects circular dependencies and provides clear error messages:

```text
Error: Circular dependency detected among validators: validator_a, validator_b
```

### Enhanced Error Messages

When validation fails, you get:

1. **Detailed field errors**: Exact fields that failed and why
2. **Applicable validators**: List of validators that can fix the issues
3. **Dependency information**: Which validators depend on others
4. **Ready-to-use commands**: Complete commands you can copy-paste

**Example error output:**

```text
Validation Error in discoveryspace 'my-space'

Fields with validation errors: 2 field(s)

Error details:
  • config.properties:
    - Extra inputs are not permitted
  • config.entitySourceIdentifier:
    - Field required

Available legacy validators:

  1. discoveryspace_properties_field_removal
     Removes the deprecated 'properties' field
     Handles: properties
     Deprecated: v0.10.1

  2. discoveryspace_entitysource_to_samplestore
     Renames 'entitySourceIdentifier' to 'sampleStoreIdentifier'
     Handles: entitySourceIdentifier
     Deprecated: v0.9.6
     Dependencies: discoveryspace_properties_field_removal

To upgrade using legacy validators:
  ado upgrade discoveryspace \
    --apply-legacy-validator discoveryspace_properties_field_removal \
    --apply-legacy-validator discoveryspace_entitysource_to_samplestore
```

### Progress Tracking

When upgrading local files, you see:

- Overall progress bar for files
- Per-file validator progress
- Real-time status updates
- Clear success/failure indicators

**Example:**

```text
Processing 3 file(s) with 2 validator(s)...

⠋ Processing file1.yaml... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 33%
  Applying validator_a... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50%

File: file1.yaml
Applied validators: validator_a, validator_b
✓ Upgraded: file1.yaml
```

### Dry-Run Mode

Preview changes without modifying files:

```bash
ado legacy upgrade --file my-resource.yaml \
  --apply-legacy-validator validator_name \
  --dry-run
```

This shows:

- Which validators would be applied
- Preview of the modified YAML
- No files are actually changed

### Backup Creation

When upgrading files in-place, backups are created automatically:

```bash
ado legacy upgrade --file my-resource.yaml \
  --apply-legacy-validator validator_name
```

Creates: `my-resource.yaml.bak`

Disable backups:

```bash
ado legacy upgrade --file my-resource.yaml \
  --apply-legacy-validator validator_name \
  --no-backup
```

### Output to Different Directory

Upgrade files to a different directory:

```bash
ado legacy upgrade \
  --file old/resource.yaml \
  --apply-legacy-validator validator_name \
  --output-dir upgraded/
```

## Available Validators

### Discovery Space Validators

#### discoveryspace_properties_field_removal

- **Handles**: `properties`
- **Description**: Removes the deprecated 'properties' field
- **Deprecated from**: v0.10.1
- **Removed from**: v1.0.0

#### discoveryspace_entitysource_to_samplestore

- **Handles**: `entitySourceIdentifier`
- **Description**: Renames 'entitySourceIdentifier' to 'sampleStoreIdentifier'
- **Deprecated from**: v0.9.6
- **Removed from**: v1.0.0

### Operation Validators

#### operation_actuators_field_removal

- **Handles**: `actuators`
- **Description**: Removes the deprecated 'actuators' field
- **Deprecated from**: v0.9.6
- **Removed from**: v1.0.0
- **See**:
  [Operation Configuration](../resources/operation.md#the-operation-configuration-yaml)

#### randomwalk_mode_to_sampler_config

- **Handles**: `mode`, `grouping`, `samplerType`
- **Description**: Migrates random_walk parameters to nested 'samplerConfig'
- **Deprecated from**: v1.0.1
- **Removed from**: v1.2
- **See**:
  [Random Walk Configuration](../operators/random-walk.md#configuring-a-randomwalk)

### Sample Store Validators

#### samplestore_kind_entitysource_to_samplestore

- **Handles**: `kind`
- **Description**: Converts resource kind from 'entitysource' to 'samplestore'
- **Deprecated from**: v0.9.6
- **Removed from**: v1.0.0

#### samplestore_module_type_entitysource_to_samplestore

- **Handles**: `moduleType`
- **Description**: Converts moduleType from 'entity_source' to 'sample_store'
- **Deprecated from**: v0.9.6
- **Removed from**: v1.0.0

#### samplestore_module_class_entitysource_to_samplestore

- **Handles**: `moduleClass`
- **Description**: Converts moduleClass from EntitySource to SampleStore naming
- **Deprecated from**: v0.9.6
- **Removed from**: v1.0.0

#### samplestore_module_name_entitysource_to_samplestore

- **Handles**: `moduleName`
- **Description**: Updates module paths from entitysource to samplestore
- **Deprecated from**: v0.9.6
- **Removed from**: v1.0.0

#### csv_constitutive_columns_migration

- **Handles**: `constitutivePropertyColumns`, `propertyMap`
- **Description**: Migrates CSV sample stores from v1 to v2 format
- **Deprecated from**: v1.3.5
- **Removed from**: v1.6.0

## Best Practices

### 1. Use `ado legacy list` First

Before upgrading, check which validators are available:

```bash
ado legacy list samplestore
```

### 2. Get Detailed Information

Use `ado legacy info` to understand what a validator does:

```bash
ado legacy info csv_constitutive_columns_migration
```

### 3. Test with Dry-Run

Always test with `--dry-run` first:

```bash
ado legacy upgrade --file my-resource.yaml \
  --apply-legacy-validator validator_name \
  --dry-run
```

### 4. Let Dependencies Auto-Resolve

Don't manually specify dependencies - the system handles it:

```bash
# Good - just specify what you need
ado upgrade samplestore --apply-legacy-validator validator_b

# Unnecessary - dependencies are automatic
ado upgrade samplestore \
  --apply-legacy-validator validator_a \
  --apply-legacy-validator validator_b
```

### 5. Keep Backups

When upgrading important files, keep the default backup behavior:

```bash
# Backups created automatically
ado legacy upgrade --file important.yaml \
  --apply-legacy-validator validator_name
```

## Troubleshooting

### Validator Not Found

If you get "Unknown legacy validator":

1. Check the validator name: `ado legacy list`
2. Ensure you're using the full identifier
3. Check for typos

### Circular Dependency Error

If you see "Circular dependency detected":

1. This indicates a bug in validator definitions
2. Report the issue with the validator names
3. Use validators individually as a workaround

### Missing Dependencies

If you see "Missing validator dependencies":

1. The validator depends on another validator that doesn't exist
2. This indicates a configuration issue
3. Report the issue with details

### Validation Still Fails

If validation fails after applying validators:

1. Check if you applied all suggested validators
2. Verify the resource format matches expectations
3. Check for additional deprecated fields
4. Use `ado legacy list` to find other applicable validators

## See Also

- [Resource Upgrade Command](../cli/upgrade.md)
- [Discovery Space Resources](../resources/discoveryspace.md)
- [Sample Store Resources](../resources/samplestore.md)
- [Operation Resources](../resources/operation.md)

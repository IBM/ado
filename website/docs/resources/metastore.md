---
status: published #Status can be draft, reviewed or published. 
---

`ado` uses a SQL database to store resource definitions and [SQLSampleStores](sample-stores.md#sqlsamplestore).
When you execute `ado` commands like `get` or `describe` they are interacting with this metastore.

By hosting a metastore on a dedicated server `ado` can be used by multiple distributed users. 

!!! info  end

    The `ado` CLI can create local metastore instances.
    Shared metastores require [separately provisioning the database server](/ado/getting-started/installing-backend-services/#using-the-distributed-mysql-backend-for-ado).

## Contexts and Projects

An instance of the metastore can host one or more `projects`. 
To access a `project` you create a `context` which contains location information, and optionally access credentials, for the `project`.

### Contexts for local projects

Local projects are stored in local metastores. 
Local metastores use SQLite.
A local metastore can hold a single project.
Hence, there is one database per local metastore instance that contains the resources associated with this project.

A context for a local metastore looks like:
```yaml
project: local-test
metadataStore:
  path: $HOME/Library/Application Support/ado/databases/local-test.db
  sslVerify: false
```

### Contexts for remote projects

Remote projects are stored in remote metastores. 
Remote metastore uses MySQL.
A remote metastore can host multiple projects.
Each project is associated with an access-controlled databases that contains the projects resources. 

!!! info end

    Everyone with access to the same remote project can see and interact with all the resources in it

A context for a remote metastore looks like: 
```yaml
project: ft-prod
metadataStore:
  host: 192.168.0.1
  password: XXXXXXXXXXX
  port: 32001
  sslVerify: false
```

## Creating a context 

To create a local or remote context in `ado`, create a file with the corresponding YAML definition (see above) and run:

```commandline
ado create context -f $YAML_FILE
```

If the context refers to a local project (a local context), a SQLite database created for the project if it doesn't exist.
If the context refers to a remote project (a remote context), the MySQL database for the project must have been created separately. 

## Listing available contexts

To see a list of contexts do 
```commandline
ado get contexts
```

This will output something like
```commandline
                  CONTEXT DEFAULT
0              finetuning        
1              ap-testing        
2       developer-testing        
3             mascots2024        
4      caikit-testharness        
5    materials-evaluation        
6                 ft-prod       *
7            unit-testing        
8       your-project-name        
9  resource-store-testing        
```

Note, the name of the context is the name of the associated project.

## The active context

To use a context you activate it with: 

```commandline
ado context $CONTEXTNAME
```

and it becomes the "Active Context". All `ado` commands that interact with the metastore, like `get`, `show`, will be directed
to the project associated with the active context.

Example:
```commandline
$ ado context materials-evaluation
Success! Now using context materials-evaluation
```

To remind yourself what the active context is run
```commandline
ado context
```

The active context is also denoted by a "*" in the output of `ado get contexts` (see output above).

> [!NOTE] 
> Although `context` appears *like* resource in `ado` e.g. you can `get` contexts, the definition is not stored in the 
> metastore, so it is purely local. 

## Deleting contexts

You can delete a context using
```commandline
ado delete context $CONTEXT_NAME
```
For remote contexts the delete operation only deletes the context yaml. 
The underlying MySQL database remains and must be deleted separately. 

For local contexts, the delete operation prompts if you want to delete the underlying SQLite database,
and thus the project. 
If you opt to delete the project, the data cannot be retrieved.
In this case, if you recreate the context a new local database will be created.

If you just delete the context, the underlying SQLite database, and hence the project data, remains.
In this case, if you recreate the context it will use the existing database. 

# Using the distributed MySQL backend for ado

This guide is intended for **administrators** who are responsible for deploying
the distributed MySQL backend for ADO or provisioning new projects on it.

## Overview

We recommend using the **Percona Operator for MySQL**, which is built on
**Percona XtraDB Cluster**, to provide a resilient and production-ready MySQL
backend. This guide assumes that this setup is being used.

## Deployment Instructions

### Kubernetes

You can deploy the Percona Operator and create a Percona XtraDB Cluster using
either of the following methods:

- [**Helm charts**](https://docs.percona.com/percona-operator-for-mysql/pxc/helm.html)
- [**Kubernetes manifest files**](https://docs.percona.com/percona-operator-for-mysql/pxc/kubectl.html)

Click on the links to follow the official Percona documentation.

### OpenShift

In OpenShift environments, the operator can be installed via **OperatorHub**
using the **Operator Lifecycle Manager (OLM)**.

Refer to the official OpenShift-specific guide here:

👉
[OpenShift Deployment Guide](https://docs.percona.com/percona-operator-for-mysql/pxc/openshift.html)

## Onboarding projects

> [!WARNING]
>
> Before proceeding make sure you have followed the steps in
> [Deployment Instructions](#deployment-instructions).

### Pre-requisites

### Software

To run the scripts in this guide you will need to have the following tools
installed:

- `kubectl`: <https://kubernetes.io/docs/tasks/tools/#kubectl>
- `mysql` client version 8: <https://formulae.brew.sh/formula/mysql-client@8.4>

### PXC-related variables

> [!NOTE]
>
> We assume that your active namespace is the one in which you installed your
> Percona XtraDB Cluster.

#### PXC Cluster name

You will need to know the name of your `pxc` cluster:

```shell
kubectl get pxc -o jsonpath='{.items[].metadata.name}'
```

**We will refer to its name as `$PXC_NAME`**.

#### PXC Cluster root credentials

You will need a highly privileged account to onboard new projects, as you will
need to create databases, users, and grant permissions. For this reason, we will
use the default `root` account.

You can retrieve its password with:

```commandline
kubectl get secret $PXC_NAME-secrets --template='{{.data.root}}' | base64 -d
```

**We will refer to this password as `$MYSQL_ADMIN_PASSWORD`**.

### Onboarding new projects

The simplest way to onboard a new project called `$PROJECT_NAME` is to use the
`forward_mysql_and_onboard_new_project.sh`. This script creates a new project in
the MySQL DB and outputs an ado context YAML that can be used to connect to it.

For example:

```shell
./forward_mysql_and_onboard_new_project.sh --admin-user root \
                      --admin-pass $MYSQL_ADMIN_PASSWORD \
                      --pxc-name $PXC_NAME \
                      --project-name $PROJECT_NAME
```

Alternatively, if you are using a hosted MySQL instance somewhere (e.g., on the
Cloud), you can use the other script: `onboard_new_project.sh`:

```shell
./onboard_new_project.sh --admin-user root \
                      --admin-pass $MYSQL_ADMIN_PASSWORD \
                      --mysql-endpoint $MYSQL_ENDPOINT \
                      --project-name $PROJECT_NAME
```

Once the project is created the context YAML can be shared with whoever needs
access to the project.

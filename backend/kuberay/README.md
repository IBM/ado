# Deploying Kuberay and creating a RayCluster

This guide is intended for users who want to run operations on an autoscaling
ray cluster deployed on kubernetes/OpenShift. Depending on cluster permissions
users may need someone with administrator privileges to install KubeRay and/or
create RayCluster objects.

## Installing KubeRay

> [!WARNING]
>
> KubeRay is included in OpenShift AI and OpenDataHub. Skip this step if they
> are already installed in your cluster.

You can install the KubeRay Operator either via Helm or Kustomize by following
the
[official documentation](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html#step-2-install-kuberay-operator).

## Deploying a RayCluster

> [!WARNING] Ray version compatibility
>
> The `ray` version set in KubeRay YAML and the one
> used in the ray head and worker containers must be compatible.
> For a more in depth guide refer to the [RayCluster configuration](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html)
> page.

We provide [an example set of values](vanilla-ray.yaml) for deploying a
RayCluster via KubeRay. To deploy it run:

``` commandline
helm upgrade --install ado-ray kuberay/ray-cluster --version 1.1.0 --values backend/kuberay/vanilla-ray.yaml
```

Feel free to customize the example file provided to suit your cluster,
such as uncommenting GPU-enabled workers.

### Enabling ado actuators to create K8s resources

#### Configuring a ServiceAccount for the RayCluster

The default Kubernetes ServiceAccount created for a RayCluster does not
have enough permissions for an ado actuator to create Kubernetes resources
(e.g., deployments, pods, services, etc.) as part of its operations.
Users are required to create a custom ServiceAccount bound to a Role with
sufficient permissions, before creating the RayCluster, to avoid
runtime errors.
Below is an example ServiceAccount bound to a Role that allows
monitoring, creating, deleting, and updating of pods, deployments and services.
It also provides access to the RayCluster resources.

<!-- markdownlint-disable-next-line code-block-style -->
```yaml
{% include "./service-account.yaml" %}
```

From the root of the ado project run the below command:

```commandline
kubectl apply -f backend/kuberay/service-account.yaml
```

This will create a ServiceAccount named `ray-deployer`.
We will reference this name later when
[deploying the RayCluster](#example-kubernetes-cluster-with-4-nodes-8-gpus-each).

More information about ServiceAccount, Role, and RoleBinding objects can be found
in the [official Kubernetes RBAC documentation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/).

#### Associating a RayCluster with the ServiceAccount

The below command shows how to set the `serviceAccountName` property for head
and worker nodes.

<!-- markdownlint-disable-next-line code-block-style -->
```bash
helm upgrade --install ado-ray kuberay/ray-cluster --version 1.1.0 \
  --values backend/kuberay/vanilla-ray-service-account.yaml \
  --set head.serviceAccountName=ray-deployer \
  --set worker.serviceAccountName=ray-deployer
```

### Best Practices for Efficient GPU Resource Utilization

To maximize the efficiency of your RayCluster and minimize GPU resource
fragmentation, we recommend the following:

- **Enable Ray Autoscaler**  
  This allows Ray to dynamically adjust the number of worker replicas based on
  task demand.

- **Use Multiple GPU Worker Variants**  
  Define several GPU worker types with varying GPU counts. This flexibility
  helps match task requirements more precisely and reduces idle GPU time.

#### Recommended Worker Configuration Strategy

Create GPU worker variants with increasing GPU counts, where each variant has
double the GPUs of the previous one. Limit each variant to a maximum of **2
replicas**, ensuring that their combined GPU usage does not exceed the capacity
of a single replica of the next larger variant.

##### Example: Kubernetes Cluster with 4 Nodes (8 GPUs Each)

Recommended worker setup:

- 2 replicas of a worker with **1 GPU**
- 2 replicas of a worker with **2 GPUs**
- 2 replicas of a worker with **4 GPUs**
- 4 replicas of a worker with **8 GPUs**

<!-- markdownlint-disable no-inline-html -->

<details>
<summary>
Example: The contents of the additionalWorkerGroups field of a RayCluster
with 4 Nodes each with 8 NVIDIA-A100-SXM4-80GB GPUs, 64 CPU cores, and 1TB memory
</summary>
<!-- markdownlint-disable MD046 -->
    ```yaml
    one-A100-80G-gpu-WG:
      replicas: 0
      minReplicas: 0
      maxReplicas: 2
      rayStartParams:
        block: 'true'
        num-gpus: '1'
        resources: '"{\"NVIDIA-A100-SXM4-80GB\": 1}"'
      containerEnv:
        - name: OMP_NUM_THREADS
          value: "1"
        - name: OPENBLAS_NUM_THREADS
          value: "1"
      lifecycle:
        preStop:
          exec:
            command: [ "/bin/sh","-c","ray stop" ]
      # securityContext: ...
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-A100-SXM4-80GB
      resources:
        limits:
          cpu: 8
          nvidia.com/gpu: 1
          memory: 100Gi
        requests:
          cpu: 8
          nvidia.com/gpu: 1
          memory: 100Gi
      # volumes: ...
      # volumeMounts: ....

    two-A100-80G-gpu-WG:
      replicas: 0
      minReplicas: 0
      maxReplicas: 2
      rayStartParams:
        block: 'true'
        num-gpus: '2'
        resources: '"{\"NVIDIA-A100-SXM4-80GB\": 2}"'
      containerEnv:
        - name: OMP_NUM_THREADS
          value: "1"
        - name: OPENBLAS_NUM_THREADS
          value: "1"
      lifecycle:
        preStop:
          exec:
            command: [ "/bin/sh","-c","ray stop" ]
      # securityContext: ...
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-A100-SXM4-80GB
      resources:
        limits:
          cpu: 15
          nvidia.com/gpu: 2
          memory: 200Gi
        requests:
          cpu: 15
          nvidia.com/gpu: 2
          memory: 200Gi
      # volumes: ...
      # volumeMounts: ....

    four-A100-80G-gpu-WG:
      replicas: 0
      minReplicas: 0
      maxReplicas: 2
      rayStartParams:
        block: 'true'
        num-gpus: '4'
        resources: '"{\"NVIDIA-A100-SXM4-80GB\": 4}"'
      containerEnv:
        - name: OMP_NUM_THREADS
          value: "1"
        - name: OPENBLAS_NUM_THREADS
          value: "1"
      lifecycle:
        preStop:
          exec:
            command: [ "/bin/sh","-c","ray stop" ]
      # securityContext: ...
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-A100-SXM4-80GB
      resources:
        limits:
          cpu: 30
          nvidia.com/gpu: 4
          memory: 400Gi
        requests:
          cpu: 30
          nvidia.com/gpu: 4
          memory: 400Gi
      # volumes: ...
      # volumeMounts: ....

    eight-A100-80G-gpu-WG:
      replicas: 0
      minReplicas: 0
      maxReplicas: 4
      rayStartParams:
        block: 'true'
        num-gpus: '8'
        resources: '"{\"NVIDIA-A100-SXM4-80GB\": 8, \"full-worker\": 1}"'
      containerEnv:
        - name: OMP_NUM_THREADS
          value: "1"
        - name: OPENBLAS_NUM_THREADS
          value: "1"
      lifecycle:
        preStop:
          exec:
            command: [ "/bin/sh","-c","ray stop" ]
      # securityContext: ...
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-A100-SXM4-80GB

      resources:
        limits:
          cpu: 60
          nvidia.com/gpu: 8
          memory: 800Gi
        requests:
          cpu: 60
          nvidia.com/gpu: 8
          memory: 800Gi
      # volumes: ...
      # volumeMounts: ....
    ```
<!-- markdownlint-enable MD046 -->
</details>
<!-- markdownlint-enable no-inline-html -->

> [!IMPORTANT] full-worker custom resource
>
> Notice that the only variant with a **full-worker** custom resource
> is the one with 8 GPUs. Some actuators, like SFTTrainer, use this
> custom resource for measurements that involve reserving an entire GPU node.

### RayClusters and SFTTrainer

> [!IMPORTANT] HuggingFace home directory
>
> If you want to run multi-node measurements with
> the SFTTrainer actuator make sure that
> all nodes in your multi-node setup have read and write access
> to your HuggingFace home directory. On Kubernetes with RayClusters,
> avoid S3-like filesystems as that is known to cause failures
> in **transformers**.
> Use a NFS or GPFS-backed PersistentVolumeClaim instead.

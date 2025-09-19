# Deploying ADO's API via Ray Serve

## Overview

This guide explains how to spin up ADO's API using Ray Serve on your **local
machine** and on a **Kuberay** cluster.

## Prerequisites

### Local deployment

Ensure you have created a virtual environment for ADO by following the
instructions in
[our development documentation](https://ibm.github.io/ado/getting-started/developing/#creating-a-development-virtual-environment)

### Kuberay deployment

Make sure you have completed the setup outlined in
[our instructions for deploying Kuberay on your Kubernetes cluster](https://ibm.github.io/ado/getting-started/installing-backend-services/#installing-kuberay).

## Instructions

### Deploying locally

Serving the API locally is very easy. Run the following command in your
terminal:

```bash
serve run orchestrator.api.rest:ado_rest_api
```

You should see output similar to this:

<!-- markdownlint-disable line-length -->
```terminaloutput
2025-09-19 11:50:39,727 INFO scripts.py:507 -- Running import path: 'orchestrator.api.rest:ado_rest_api'.
2025-09-19 11:50:45,496 INFO worker.py:1942 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265
(ProxyActor pid=98735) INFO 2025-09-19 11:50:48,686 proxy 127.0.0.1 -- Proxy starting on node 05f52ce870e3943ff5ec646472a93f5b552fd48ad45cdd8286569db1 (HTTP port: 8000).
(ProxyActor pid=98735) INFO 2025-09-19 11:50:48,801 proxy 127.0.0.1 -- Got updated endpoints: {}.
INFO 2025-09-19 11:50:48,825 serve 98612 -- Started Serve in namespace "serve".
INFO 2025-09-19 11:50:48,852 serve 98612 -- Connecting to existing Serve app in namespace "serve". New http options will not be applied.
(ServeController pid=98727) INFO 2025-09-19 11:50:49,037 controller 98727 -- Deploying new version of Deployment(name='AdoRESTApi', app='default') (initial target replicas: 1).
(ProxyActor pid=98735) INFO 2025-09-19 11:50:49,078 proxy 127.0.0.1 -- Got updated endpoints: {Deployment(name='AdoRESTApi', app='default'): EndpointInfo(route='/', app_is_cross_language=False)}.
(ProxyActor pid=98735) INFO 2025-09-19 11:50:49,098 proxy 127.0.0.1 -- Started <ray.serve._private.router.SharedRouterLongPollClient object at 0x11531f730>.
(ServeController pid=98727) INFO 2025-09-19 11:50:49,177 controller 98727 -- Adding 1 replica to Deployment(name='AdoRESTApi', app='default').
INFO 2025-09-19 11:50:50,096 serve 98612 -- Application 'default' is ready at http://127.0.0.1:8000/.
```
<!-- markdownlint-enable line-length -->

Once you see the final line, the API is running. Open the interactive OpenAPI
documentation at <http://127.0.0.1:8000/docs> or the ReDoc version at
<http://127.0.0.1:8000/redoc>.

## Deploying on Kuberay

Ray Serve applications are deployed on Kuberay via
[`RayService`s](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice.html).
An example RayService is provided:

```yaml
{% include-markdown "./ado-api-rayserve.yaml" %}
```

From the root of the ado project directory, you can deploy it with:

```bash
kubectl apply -f backend/api/ado-api-rayserve.yaml
```

Kuberay will automatically create a service for the Serve endpoint called
`${RAY_SERVICE_NAME}-serve-svc`. In the case of our example, this will be
`ado-api-serve-svc`.

> [!TIP]
>
> For ease of use, we suggest exposing the service using either a Route (on
> OpenShift), a LoadBalancer service or an Ingress. **Make sure you take
> appropriate security measures to protect the endpoint**.

You can access it via port-forward using:

```bash
kubectl port-forward svc/ado-api-serve-svc 8000:8000
```

You can then navigate to the interactive OpenAPI documentation at
<http://127.0.0.1:8000/docs> or the ReDoc version at
<http://127.0.0.1:8000/redoc>.

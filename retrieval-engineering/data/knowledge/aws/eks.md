# Amazon EKS

Amazon Elastic Kubernetes Service (EKS) is a managed control plane for
running Kubernetes on AWS. AWS operates the API server, etcd, and control
plane scaling and patching across multiple Availability Zones; you bring
worker nodes (EC2, Fargate, or a mix) that join the cluster and run pods.

## Authentication

EKS does not have its own user database. Cluster authentication is
delegated to AWS Identity and Access Management (IAM): a client calls the
EKS API using AWS credentials (via the AWS CLI's `aws eks get-token` or an
equivalent SDK call), which produces a short-lived, signed token. The
Kubernetes API server validates that token against IAM through the
`aws-iam-authenticator` (or the built-in EKS authenticator) and maps the
calling IAM principal to a Kubernetes username and group via the
`aws-auth` ConfigMap (or, in newer clusters, EKS access entries).

## Authorization

Once a request is authenticated and mapped to a Kubernetes identity,
standard Kubernetes RBAC takes over: the mapped user or group must have
a RoleBinding or ClusterRoleBinding granting it permission to perform the
requested action on the requested resource. This two-layer model --
IAM answers "who are you," RBAC answers "what can you do" -- is the core
mechanic to understand when debugging EKS access issues.

## Networking

Pods get IP addresses directly from the VPC subnet via the Amazon VPC CNI
plugin, so pod-to-pod and pod-to-AWS-service traffic can be controlled
with ordinary VPC security groups and network ACLs, not just Kubernetes
NetworkPolicies.

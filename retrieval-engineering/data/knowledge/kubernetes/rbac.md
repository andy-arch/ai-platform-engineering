# Kubernetes RBAC

Role-Based Access Control (RBAC) governs what an already-authenticated
user, group, or service account is allowed to do against the Kubernetes
API. Authentication answers "who are you"; RBAC answers "what can you
do" -- the same two-layer separation used by AWS IAM.

## Roles and ClusterRoles

A `Role` grants a set of permissions (verbs like `get`, `list`, `watch`,
`create`, `delete` on resources like `pods` or `deployments`) scoped to a
single namespace. A `ClusterRole` grants the same kind of permissions but
either cluster-wide, or reusable across namespaces, or for cluster-scoped
resources (like `nodes`) that don't belong to any namespace at all.

## Bindings

A Role or ClusterRole by itself grants nothing -- it has to be bound to a
subject. A `RoleBinding` grants a Role's (or a ClusterRole's) permissions
to a user, group, or service account within one namespace. A
`ClusterRoleBinding` grants a ClusterRole's permissions across the entire
cluster. A common pattern is binding the same ClusterRole (e.g.
"view-only") via separate RoleBindings in multiple namespaces, reusing
the permission definition while keeping the grant itself
namespace-scoped.

## Least privilege in practice

Because permissions are additive (there's no explicit Deny in Kubernetes
RBAC, unlike IAM), the practical way to restrict access is to grant only
narrowly-scoped Roles rather than reusing broad built-in ClusterRoles
like `cluster-admin`, and to prefer namespaced RoleBindings over
cluster-wide bindings wherever the workload allows it.

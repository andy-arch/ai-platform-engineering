# Kubernetes Deployments

A Deployment is the standard Kubernetes object for running a stateless
application. It doesn't manage pods directly -- it manages a ReplicaSet,
which in turn manages the actual pods -- but in normal use you only ever
interact with the Deployment.

## Replica management

A Deployment's spec declares a desired pod template and a replica count.
The controller continuously reconciles reality toward that desired state:
if a pod crashes or its node fails, the ReplicaSet notices the pod count
has dropped below the desired number and creates a replacement. This
reconciliation loop -- observe actual state, compare to desired state,
act to close the gap -- is the pattern every Kubernetes controller
follows, not just Deployments.

## Rolling updates

When you change a Deployment's pod template (e.g. a new container image),
the default `RollingUpdate` strategy creates a new ReplicaSet and
gradually shifts pods from the old ReplicaSet to the new one, controlled
by `maxSurge` (how many extra pods can exist temporarily above the
desired count) and `maxUnavailable` (how many pods can be down at once).
The old ReplicaSet is scaled to zero, not deleted, which is what makes
`kubectl rollout undo` fast -- it just scales the previous ReplicaSet
back up.

## Readiness and health

A rolling update only proceeds to replace the next batch of pods once the
new pods report Ready via their readiness probe, which is what prevents a
bad rollout from taking down all replicas at once.

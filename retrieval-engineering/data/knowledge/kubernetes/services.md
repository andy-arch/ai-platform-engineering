# Kubernetes Services

Pods are ephemeral -- they get a new IP address every time they're
recreated -- so nothing should ever talk to a pod's IP directly. A
Service gives a stable name and IP address that routes to whichever pods
currently match a label selector, regardless of how many times those
pods are replaced.

## How routing works

A Service doesn't proxy traffic through a central process by default.
Each node runs `kube-proxy`, which watches the Service and its matching
Endpoints (the set of pod IPs currently selected) and programs local
routing rules (via iptables or IPVS) so traffic to the Service's virtual
IP is transparently rewritten to a chosen pod's real IP, on the node that
received the packet. This means Service routing has no single point of
failure and no extra network hop through a broker process.

## Service types

`ClusterIP` (the default) is reachable only from inside the cluster.
`NodePort` additionally exposes the Service on a static port on every
node's IP. `LoadBalancer` provisions an external cloud load balancer (an
AWS NLB/ALB on EKS) pointing at the Service. `ExternalName` is a special
case that returns a DNS CNAME instead of proxying traffic at all, useful
for referring to a resource outside the cluster by an in-cluster name.

## DNS

Every Service gets a DNS name automatically via the cluster's DNS add-on
(CoreDNS), of the form `<service-name>.<namespace>.svc.cluster.local`,
which is how most in-cluster service-to-service calls are addressed in
practice rather than by IP.

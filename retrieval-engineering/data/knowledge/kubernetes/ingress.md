# Kubernetes Ingress

Ingress is the standard Kubernetes API for routing external HTTP(S)
traffic into the cluster based on hostname and URL path, letting many
Services share a single external load balancer instead of each needing
its own.

## Ingress vs. Ingress Controller

An Ingress object is just a declarative set of routing rules ("requests
for `api.example.com/v1` go to Service `api-v1`") -- by itself it does
nothing. An Ingress Controller (NGINX Ingress Controller, AWS Load
Balancer Controller, Traefik, etc.) is what actually watches Ingress
objects and configures a real proxy or cloud load balancer to implement
those rules. A cluster with no Ingress Controller installed will accept
Ingress objects but never route any traffic for them.

## How a request flows

On EKS with the AWS Load Balancer Controller, creating an Ingress
provisions an Application Load Balancer (ALB) in the VPC. The ALB
terminates TLS (if configured) and forwards each request, based on the
Ingress rules, to the target Service, which in turn load-balances across
the matching pods.

## TLS termination

TLS certificates for an Ingress are typically supplied via a Kubernetes
Secret referenced in the Ingress spec, or provisioned automatically
through an integration like AWS Certificate Manager (for ALB) or
cert-manager (for self-managed certificates via Let's Encrypt or another
issuer). TLS is normally terminated at the load balancer or ingress
controller, not inside the application pod.

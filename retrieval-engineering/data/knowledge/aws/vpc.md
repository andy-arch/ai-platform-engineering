# Amazon VPC

A Virtual Private Cloud (VPC) is an isolated, logically-separated network
within AWS where you launch resources such as EC2 instances, EKS worker
nodes, and RDS databases. Each VPC has its own private IP address range
(defined as a CIDR block, e.g. `10.0.0.0/16`) and is fully isolated from
other customers' VPCs by default.

## Subnets and routing

A VPC is divided into subnets, each pinned to a single Availability Zone.
A subnet is "public" if its route table sends traffic destined outside
the VPC to an Internet Gateway, and "private" if it instead routes
outbound traffic through a NAT Gateway (so resources can reach the
internet but can't be reached directly from it). Route tables, not
subnet labels, are what actually determine reachability.

## Security boundaries

Two independent layers control traffic: security groups are stateful
firewalls attached to individual resources (e.g. an EC2 instance or an
EKS pod's ENI), evaluated only as allow-rules; network ACLs are stateless
firewalls attached to a subnet, evaluated as an ordered list of allow and
deny rules. Because they're independent, traffic must be permitted by
both to succeed.

## Connectivity between VPCs

VPC peering connects two VPCs directly (non-transitive -- peering A-B and
B-C does not let A reach C). AWS Transit Gateway is the alternative for
connecting many VPCs (and on-premises networks) through a single
transitive routing hub, which scales far better than a full mesh of
peering connections.

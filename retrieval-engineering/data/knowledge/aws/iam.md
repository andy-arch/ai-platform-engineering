# AWS IAM

AWS Identity and Access Management (IAM) is the service that controls who
can do what across an AWS account. It has no data plane of its own --
every other AWS service asks IAM "is this request allowed?" before acting.

## Principals

A principal is an identity that can make a request: an IAM user (a
long-lived human or application identity with its own credentials), an
IAM role (a temporary identity assumed by a user, service, or another
AWS account -- no long-lived credentials, just short-lived STS tokens),
or an AWS service acting on your behalf.

## Policies

Permissions are granted through JSON policy documents attached to a
principal (or, for resource policies, attached to the resource itself,
such as an S3 bucket policy). A policy statement specifies an `Effect`
(Allow or Deny), one or more `Action`s (e.g. `s3:GetObject`), and one or
more `Resource`s (ARNs) the action applies to, optionally narrowed by a
`Condition`. An explicit Deny always wins over any Allow, even one
granted by a different policy.

## Roles vs. long-lived credentials

Production systems should prefer IAM roles over long-lived access keys
wherever possible: an EC2 instance, EKS pod (via IAM Roles for Service
Accounts), or Lambda function can assume a role and receive automatically
rotated, short-lived credentials, removing the need to store and rotate
static secrets.

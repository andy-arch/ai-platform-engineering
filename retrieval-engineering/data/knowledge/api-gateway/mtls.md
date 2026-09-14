# Mutual TLS (mTLS)

Ordinary TLS only proves the server's identity to the client -- the
client verifies the server's certificate, but the server has no
cryptographic proof of who the client is. Mutual TLS adds a second
certificate exchange in the other direction, so both sides authenticate
each other during the same handshake.

## How the handshake differs

In a standard TLS handshake, the server presents a certificate signed by
a trusted Certificate Authority (CA), and the client verifies it. In an
mTLS handshake, the server additionally requests a certificate from the
client; the client presents its own certificate, and the server verifies
it against a CA it trusts (often a private, internal CA rather than a
public one, since mTLS is typically used for service-to-service traffic
rather than public-facing clients). Only if both verifications succeed
does the connection proceed.

## Where it's used

mTLS is common inside a service mesh (e.g. Istio, Linkerd) to authenticate
every service-to-service call within a cluster, and at an API gateway
protecting a B2B or partner API where the caller is a known, specific
system rather than an arbitrary end user. It's rarely used for
consumer-facing APIs, since distributing and rotating client certificates
to end-user devices is operationally heavy compared to a bearer token.

## Certificate rotation

Because mTLS relies on certificates rather than long-lived shared
secrets, its main operational burden is rotation: certificates have
expiry dates, and a service mesh's sidecar proxies typically automate
issuance and rotation so individual services never handle private keys
directly.

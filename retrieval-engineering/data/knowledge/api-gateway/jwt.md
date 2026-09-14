# JWT Authentication

A JSON Web Token (JWT) is a compact, self-contained token format used to
prove a user's identity and claims to an API without the API having to
call back to a central session store on every request.

## Structure

A JWT has three base64url-encoded parts separated by dots: a header
(describing the signing algorithm), a payload (the claims -- e.g.
`sub` for subject, `exp` for expiry, and any application-specific
claims like roles or scopes), and a signature. The signature is computed
over the header and payload using a secret (for symmetric algorithms like
HS256) or a private key (for asymmetric algorithms like RS256).

## Why an API gateway validates it

At an API gateway, JWT validation means: verify the signature against the
known public key or shared secret, and check standard claims such as
`exp` (has it expired), `nbf` (not valid before), and `aud`/`iss`
(is this token intended for this API, issued by a trusted issuer). If all
checks pass, the gateway trusts the claims inside the token without
querying the identity provider again -- this is what makes JWTs
"stateless": verification is pure computation, not a database lookup.

## Common pitfalls

Using a symmetric algorithm (HS256) with a secret embedded in a client
application lets that client forge tokens, since the same secret both
signs and verifies; asymmetric algorithms (RS256) avoid this because the
private signing key never leaves the issuer. Failing to check `exp` and
`aud` is the most common real-world JWT validation bug -- a token that
signature-verifies correctly but was issued for a different service or
has already expired must still be rejected.

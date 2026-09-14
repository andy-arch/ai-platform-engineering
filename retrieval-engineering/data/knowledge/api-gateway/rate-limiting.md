# Rate Limiting

Rate limiting caps how many requests a client can make in a given time
window, protecting an API from being overwhelmed by a single misbehaving
or malicious client and giving predictable capacity to every other
client.

## Common algorithms

The token bucket algorithm gives each client a bucket that refills with
tokens at a fixed rate up to a maximum capacity; each request consumes
one token, and a request is rejected if the bucket is empty. This allows
short bursts (as long as the bucket has tokens saved up) while enforcing
a steady average rate. The sliding window algorithm instead counts
requests in a rolling time window, avoiding the burst-at-the-boundary
problem of a naive fixed window (where a client could send double its
allowed rate by timing requests just before and after a window reset).

## Where it's enforced

At an API gateway, rate limiting is typically applied per identity (per
API key, per authenticated user, or per IP address for unauthenticated
traffic) rather than globally, so one client hitting its limit doesn't
affect others. Limits are often tiered by client type -- a paying
customer's API key might get a much higher limit than the default for
unauthenticated requests.

## Response contract

A rate-limited request should be rejected with HTTP 429 (Too Many
Requests), and well-behaved APIs include a `Retry-After` header (or
`X-RateLimit-*` headers indicating remaining quota and reset time) so the
client can back off intelligently instead of retrying immediately and
making the problem worse.

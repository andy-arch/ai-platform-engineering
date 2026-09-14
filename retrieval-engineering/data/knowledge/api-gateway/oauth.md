# OAuth 2.0

OAuth 2.0 is an authorization framework: it lets a user grant a third
party application limited access to their resources on another service,
without sharing their password with that application. It answers "what
is this application allowed to do on my behalf," which is a different
question from authentication ("who is this user"), though the two are
often used together (OpenID Connect layers authentication on top of
OAuth 2.0).

## Authorization code flow

The most common flow for a web application: the app redirects the user
to the authorization server's login page; the user authenticates and
approves the requested scopes; the authorization server redirects back to
the app with a short-lived authorization code; the app's backend
exchanges that code (plus a client secret, server-side) for an access
token and refresh token. The access token is what's actually sent to the
resource API on each request, typically as a Bearer token.

## Tokens

The access token is short-lived and is what an API gateway checks on
every request (often, in practice, a JWT -- see the JWT document for how
that validation works). The refresh token is longer-lived and is used
only to obtain a new access token without forcing the user to log in
again, and should be stored and transmitted more carefully since it
represents longer-term access.

## Client types

A "confidential client" (a server-side app) can safely hold a client
secret; a "public client" (a mobile app or single-page browser app)
cannot, and instead uses PKCE (Proof Key for Code Exchange) to prevent
the authorization code from being intercepted and redeemed by an
attacker.

"""Auth provider factory."""

from .auth_provider import AuthProvider, NoAuthProvider


def get_auth_provider(provider_name: str) -> AuthProvider:
    """Return an auth provider instance based on configuration."""
    if provider_name.lower() in {"abstract", "none", "noauth"}:
        return NoAuthProvider()

    # Placeholder for real providers (Google, Azure, GCP)
    return NoAuthProvider()

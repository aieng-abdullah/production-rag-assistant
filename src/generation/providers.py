"""LLM provider registry with failover chain."""

from dataclasses import dataclass

from loguru import logger


@dataclass
class Provider:
    name: str   # "groq" | "anthropic" | "openai"
    api_key: str
    model: str


@dataclass
class ProviderOverrides:
    """Per-request provider settings (e.g. sidebar-entered keys).

    Passed as function arguments through the call chain — never written
    to Config. Empty fields fall back to Config values.
    """

    groq_model: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    openai_api_key: str = ""
    openai_model: str = ""


def build_provider_chain(overrides: ProviderOverrides | None = None) -> list[Provider]:
    """Return providers that have API keys configured, in failover order.

    Order: Groq (free) → Anthropic → OpenAI.
    Only includes providers whose API key is non-empty.
    Non-empty `overrides` fields take precedence over Config values.
    """
    from src.config import Config

    chain: list[Provider] = []

    if Config.GROQ_API_KEY:
        groq_model = (overrides.groq_model if overrides else "") or Config.GROQ_MODEL
        chain.append(Provider("groq", Config.GROQ_API_KEY, groq_model))

    anthropic_key = (overrides.anthropic_api_key if overrides else "") or Config.ANTHROPIC_API_KEY
    anthropic_model = (overrides.anthropic_model if overrides else "") or Config.ANTHROPIC_MODEL
    if anthropic_key:
        chain.append(Provider("anthropic", anthropic_key, anthropic_model))

    openai_key = (overrides.openai_api_key if overrides else "") or Config.OPENAI_API_KEY
    openai_model = (overrides.openai_model if overrides else "") or Config.OPENAI_MODEL
    if openai_key:
        chain.append(Provider("openai", openai_key, openai_model))

    if not chain:
        raise RuntimeError(
            "No LLM providers configured. Set at least one of "
            "GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
        )

    logger.debug(f"Provider chain: {[p.name for p in chain]}")
    return chain


def create_langchain_client(provider: Provider):
    """Instantiate the LangChain chat model for the given provider.

    Imports are local so users only need to install the package for
    providers they actually use.
    """
    if provider.name == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=provider.api_key, model=provider.model)

    if provider.name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Install it with: pip install langchain-anthropic"
            )
        return ChatAnthropic(api_key=provider.api_key, model=provider.model)

    if provider.name == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed. "
                "Install it with: pip install langchain-openai"
            )
        return ChatOpenAI(api_key=provider.api_key, model=provider.model)

    raise ValueError(f"Unknown provider: {provider.name}")

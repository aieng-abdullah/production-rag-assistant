"""LLM provider registry with failover chain."""

from dataclasses import dataclass

from loguru import logger


@dataclass
class Provider:
    name: str   # "groq" | "anthropic" | "openai"
    api_key: str
    model: str


def build_provider_chain() -> list[Provider]:
    """Return providers that have API keys configured, in failover order.

    Order: Groq (free) → Anthropic → OpenAI.
    Only includes providers whose API key is non-empty.
    """
    from src.config import Config

    chain: list[Provider] = []

    if Config.GROQ_API_KEY:
        chain.append(Provider("groq", Config.GROQ_API_KEY, Config.GROQ_MODEL))
    if Config.ANTHROPIC_API_KEY:
        chain.append(Provider("anthropic", Config.ANTHROPIC_API_KEY, Config.ANTHROPIC_MODEL))
    if Config.OPENAI_API_KEY:
        chain.append(Provider("openai", Config.OPENAI_API_KEY, Config.OPENAI_MODEL))

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

"""
Factories for the two models the application needs.

Both are cached singletons: the embedding model costs ~130 MB of RAM and several
seconds to load, and the chat client holds a connection pool — rebuilding either
per request would dominate latency.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


# BGE models are instruction-tuned: they expect *queries* (not documents) to be
# prefixed with a short instruction, and omitting it measurably hurts recall.
# MiniLM and most other sentence-transformers models are not, and prefixing them
# actively degrades results — so the prefix is applied by model family, not
# unconditionally.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# OpenRouter rejects a "models" routing array longer than this.
_MAX_ROUTING_CHAIN = 3


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the local sentence-transformers embedding model.

    Runs entirely on CPU with no API key.

    Two model families need different handling:

    * **BGE** models are instruction-tuned, so queries must carry a prefix.
      That behaviour lives in a dedicated class (``HuggingFaceBgeEmbeddings``);
      the plain ``HuggingFaceEmbeddings`` class rejects ``query_instruction``
      outright, so the choice of class *is* the configuration.
    * **MiniLM** and most other models take no prefix, and adding one degrades
      retrieval.

    ``normalize_embeddings=True`` applies to both: it turns FAISS's
    inner-product search into exact cosine similarity, which is what these
    models are trained for and what puts scores in a comparable [0, 1] range.
    """
    model_name = settings.embedding_model
    is_bge = "bge" in model_name.lower()
    shared = {
        "model_name": model_name,
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
    }

    logger.info("Loading embedding model %s (downloaded once, then cached)", model_name)

    if is_bge:
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            return HuggingFaceBgeEmbeddings(**shared, query_instruction=_BGE_QUERY_INSTRUCTION)
        except Exception as exc:
            # Losing the prefix costs some recall but still works correctly.
            logger.warning("BGE embedding class unavailable (%s); using the plain one", exc)

    return HuggingFaceEmbeddings(**shared)


def build_llm(*, streaming: bool = False, temperature: float | None = None,
              tags: list[str] | None = None, provider: str | None = None) -> ChatOpenAI:
    """Construct a chat model pointed at OpenRouter.

    Requesty, OpenRouter and OpenAI are all wire-compatible with the OpenAI
    API, so ``ChatOpenAI`` works unchanged once ``base_url`` and the key are
    pointed at the configured provider.

    Args:
        streaming: enable token-by-token streaming (used by the answer node).
        temperature: overrides the configured default; extraction uses 0.0.
        tags: LangChain run tags. The chat route filters LangGraph's event
            stream on these, so only the *final answer* model emits tokens to
            the browser — intermediate LLM calls stay silent.
        provider: build against this provider instead of the configured one.
            Used to assemble the cross-provider fallback chain.
    """
    provider = (provider or settings.llm_provider).lower()
    api_key = settings.key_for(provider)
    model = settings.model_for(provider)
    base_url = settings.base_url_for(provider)
    if not api_key:
        raise RuntimeError(
            f"No API key for provider '{provider}'. "
            f"Set {provider.upper()}_API_KEY (or LLM_API_KEY) in .env"
        )

    # OpenRouter-specific routing options, passed straight through in the request
    # body. ``models`` is a fallback chain: if the primary model returns an error
    # (notably a 429 from a ':free' model's shared upstream pool), OpenRouter
    # transparently retries the next one instead of failing the request.
    #
    # This matters a lot on the free tier. Free models share a rate-limit pool
    # across all OpenRouter users, so a 429 says nothing about *our* usage and
    # can arrive at any moment. Client-side retries do not help — the pool is
    # still exhausted — but a different model usually works immediately.
    # Fallback routing is an OpenRouter extension. Sending it to another
    # provider is at best ignored and at worst a 400, so it is gated.
    extra_body: dict = {}
    if provider == "openrouter" and settings.openrouter_fallback_models:
        fallbacks = [
            m.strip() for m in settings.openrouter_fallback_models.split(",") if m.strip()
        ]
        if fallbacks:
            chain = [model, *fallbacks]
            # OpenRouter rejects a routing chain longer than three entries, so
            # keep the primary plus the two best alternates and drop the rest
            # rather than letting the request 400.
            extra_body["models"] = chain[:_MAX_ROUTING_CHAIN]

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
        timeout=180,
        max_retries=4,
        tags=tags or [],
        extra_body=extra_body or None,
        # Attribution headers. OpenRouter shows these on its dashboard;
        # other providers ignore them harmlessly.
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "GraphRAG Emirates NBD",
        },
    )


def _provider_chain() -> list[str]:
    """Providers to try, in order, that actually have a key configured.

    The configured provider goes first; the others become fallbacks. A
    provider with no key is dropped rather than left to fail at call time.
    """
    primary = settings.llm_provider.lower()
    ordered = [primary] + [p for p in settings.fallback_provider_list if p != primary]
    return [p for p in ordered if settings.key_for(p)]


def build_with_fallbacks(*, streaming: bool = False, temperature: float | None = None,
                         tags: list[str] | None = None):
    """Build the model with cross-provider failover.

    Why this exists: OpenRouter's ``models`` parameter only fails over between
    models *inside OpenRouter*. It cannot help when OpenRouter itself returns
    429 because the account's daily free quota is spent, or when Requesty
    returns 402 because the balance is empty — both of which happened while
    building this. Those are provider-level outages, and surviving them needs
    a second provider.

    LangChain's ``with_fallbacks`` wraps the primary runnable so that any
    exception moves to the next one transparently. Callers see one runnable and
    do not know or care which provider answered.

    Returns:
        The primary model when only one provider is configured, otherwise a
        runnable that tries each in turn.
    """
    providers = _provider_chain()
    if not providers:
        raise RuntimeError(
            "No LLM provider has an API key. Set REQUESTY_API_KEY or "
            "OPENROUTER_API_KEY in .env"
        )

    models = [
        build_llm(streaming=streaming, temperature=temperature, tags=tags, provider=p)
        for p in providers
    ]
    if len(models) == 1:
        logger.info("LLM provider: %s (no fallback configured)", providers[0])
        return models[0]

    logger.info("LLM providers: %s -> %s", providers[0], ", ".join(providers[1:]))
    return models[0].with_fallbacks(models[1:])


def build_structured_with_fallbacks(schema, *, temperature: float = 0.0,
                                    tags: list[str] | None = None):
    """Same failover, for structured output.

    ``with_structured_output`` has to be applied to each concrete model
    *before* they are chained: the wrapper returned by ``with_fallbacks`` is a
    plain runnable and no longer exposes the method.
    """
    providers = _provider_chain()
    if not providers:
        raise RuntimeError("No LLM provider has an API key.")

    structured = [
        build_llm(streaming=False, temperature=temperature, tags=tags, provider=p)
        .with_structured_output(schema)
        for p in providers
    ]
    return structured[0] if len(structured) == 1 else structured[0].with_fallbacks(structured[1:])


@lru_cache(maxsize=1)
def get_extraction_llm():
    """Deterministic, non-streaming model used for entity/relationship extraction.

    Temperature is pinned to 0 so the same chunk yields the same triples across
    runs — important because ingestion is idempotent by design.
    """
    return build_with_fallbacks(streaming=False, temperature=0.0, tags=["extraction"])


def get_structured_extractor(schema):
    """Structured-output extractor with cross-provider failover.

    Not cached on the schema object itself because schemas are unhashable in
    the general case; the underlying clients are cheap to rebuild and hold no
    per-call state.
    """
    return build_structured_with_fallbacks(schema, temperature=0.0, tags=["extraction"])


@lru_cache(maxsize=1)
def get_answer_llm():
    """Streaming model that writes the grounded answer.

    Tagged ``final_answer`` so the SSE layer can forward only these tokens.
    The tag is set on every provider in the chain, so streaming keeps working
    whichever one ends up answering.
    """
    return build_with_fallbacks(streaming=True, tags=["final_answer"])

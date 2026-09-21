"""
Central configuration for the GraphRAG application.

Every tunable (credentials, model names, chunk sizes, retrieval widths) is read
from environment variables / the .env file exactly once, at import time, and
exposed through a single frozen ``settings`` object. Nothing else in the code
base is allowed to call ``os.getenv`` — that keeps configuration auditable and
makes the whole app reconfigurable without touching Python.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_project_root() -> Path:
    """Locate the repository root by walking up for a marker file.

    Counting ``.parent`` levels is brittle -- moving this module one package
    deeper silently changes what "root" means, and every relative path
    (the PDF, the FAISS index, .env) resolves somewhere wrong. Searching for a
    marker instead makes the answer independent of where this file lives.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "requirements.txt").exists() or (candidate / ".git").exists():
            return candidate
    # Fallback for an installed package with no repo markers: app/core -> app -> root
    return here.parents[2]


# Used to resolve every relative path, so the app behaves identically whether it
# is launched from the project root, from ui/, or from inside a container.
PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """Typed, validated view of the environment.

    ``AliasChoices`` lets each field accept several spellings. That matters here
    because the original .env shipped with lowercase keys (``neo4jusername``,
    ``openrouter_key``); accepting both means the app keeps working if someone
    restores an older .env.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # NEO4J_URI and neo4j_uri are treated as the same key
        extra="ignore",         # unknown .env keys are tolerated, not fatal
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings,
                                   env_settings, dotenv_settings, file_secret_settings):
        """Let .env win over process environment variables.

        Pydantic's default order puts OS environment variables ahead of the
        .env file. That is wrong for this application, and the failure mode is
        nasty: `docker compose` exports OPENROUTER_MODEL, DEFAULT_PDF_PATH and
        friends into the shell, and every later run in that shell then quietly
        ignores edits to .env. Config changes appear to do nothing, and the app
        keeps using a model or a document you thought you had replaced.

        Inside a container the reasoning inverts, but the outcome is the same:
        no .env file is copied into the image, so dotenv_settings is empty and
        the environment variables compose injects are used as intended.

        Init arguments still take precedence, so tests can override anything.
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    # ---------------------------------------------------------------- Neo4j --
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        validation_alias=AliasChoices("NEO4J_URI", "neo4j_uri", "neo4juri"),
        description="Bolt endpoint. 'bolt://neo4j:7687' inside docker compose.",
    )
    neo4j_username: str = Field(
        default="neo4j",
        validation_alias=AliasChoices("NEO4J_USERNAME", "neo4jusername", "neo4j_user"),
    )
    neo4j_password: str = Field(
        default="",
        validation_alias=AliasChoices("NEO4J_PASSWORD", "neo4jpassword", "neo4j_pass"),
    )
    neo4j_database: str = Field(default="neo4j", validation_alias=AliasChoices("NEO4J_DATABASE"))

    # ------------------------------------------------------------------ LLM --
    # The provider is any OpenAI-compatible endpoint -- Requesty, OpenRouter,
    # OpenAI itself, or a local server. Only the base URL, key and model name
    # change; no code path is provider-specific.
    llm_provider: str = Field(
        default="requesty",
        validation_alias=AliasChoices("LLM_PROVIDER"),
        description="'requesty' | 'openrouter' | 'openai' | 'custom'. Selects "
                    "which key and base URL to use when they are not set explicitly.",
    )
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY"),
        description="Overrides the provider-specific key when set.",
    )
    llm_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL"),
        description="Overrides the provider-specific base URL when set.",
    )

    # ---- Provider credentials ----
    requesty_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("REQUESTY_API_KEY", "requesty_api_key", "requesty_key"),
    )
    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_key", "openrouter_apikey"),
    )
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))

    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL"),
    )
    requesty_base_url: str = Field(
        default="https://router.requesty.ai/v1",
        validation_alias=AliasChoices("REQUESTY_BASE_URL"),
    )

    openrouter_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        validation_alias=AliasChoices("LLM_MODEL", "OPENROUTER_MODEL", "llm_model"),
        description="Must support tool calling: graph extraction relies on "
                    "structured output.",
    )
    openrouter_fallback_models: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_FALLBACK_MODELS", "OPENROUTER_FALLBACK_MODELS"),
        description="Comma-separated models to fall back to on error. Only "
                    "OpenRouter honours this; ignored by other providers.",
    )
    llm_temperature: float = Field(default=0.0, validation_alias=AliasChoices("LLM_TEMPERATURE"))
    llm_max_tokens: int = Field(
        default=1000,
        validation_alias=AliasChoices("LLM_MAX_TOKENS"),
        description="Cap on generated tokens. Worth setting explicitly: with no "
                    "cap, OpenRouter prices a request against the model's full "
                    "context window and refuses it on a low balance, even though "
                    "the actual answer is short.",
    )

    llm_fallback_providers: str = Field(
        default="openrouter,requesty",
        validation_alias=AliasChoices("LLM_FALLBACK_PROVIDERS"),
        description="Comma-separated providers to fail over to, in order, when "
                    "the primary one errors. Providers without a key are skipped.",
    )

    # ---- Per-provider model overrides ----
    # The same model is often published under different ids. OpenRouter needs a
    # ":free" suffix for its free tier; Requesty and OpenAI do not use one. A
    # single LLM_MODEL would therefore break the moment failover kicked in, so
    # each provider may name the model its own way. Empty means "use LLM_MODEL".
    llm_model_requesty: str = Field(default="", validation_alias=AliasChoices("LLM_MODEL_REQUESTY"))
    llm_model_openrouter: str = Field(default="", validation_alias=AliasChoices("LLM_MODEL_OPENROUTER"))
    llm_model_openai: str = Field(default="", validation_alias=AliasChoices("LLM_MODEL_OPENAI"))

    def model_for(self, provider: str) -> str:
        """Model id to use with one provider, falling back to LLM_MODEL."""
        override = {
            "requesty": self.llm_model_requesty,
            "openrouter": self.llm_model_openrouter,
            "openai": self.llm_model_openai,
        }.get(provider.lower(), "")
        return override or self.openrouter_model

    def key_for(self, provider: str) -> str:
        """API key for one provider. An explicit LLM_API_KEY overrides all."""
        if self.llm_api_key:
            return self.llm_api_key
        return {
            "requesty": self.requesty_api_key,
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
        }.get(provider.lower(), "")

    def base_url_for(self, provider: str) -> str:
        """Base URL for one provider. An explicit LLM_BASE_URL overrides all."""
        if self.llm_base_url:
            return self.llm_base_url
        return {
            "requesty": self.requesty_base_url,
            "openrouter": self.openrouter_base_url,
            "openai": "https://api.openai.com/v1",
        }.get(provider.lower(), self.openrouter_base_url)

    @property
    def fallback_provider_list(self) -> list[str]:
        """Fallback providers, parsed and normalised."""
        return [p.strip().lower() for p in self.llm_fallback_providers.split(",") if p.strip()]

    @property
    def active_api_key(self) -> str:
        """Key for the configured provider."""
        return self.key_for(self.llm_provider)

    @property
    def active_base_url(self) -> str:
        """Base URL for the configured provider."""
        return self.base_url_for(self.llm_provider)

    # ---- Extraction throughput ----
    # OpenRouter ':free' models permit only a few requests per minute. Exceeding
    # that produces 429s and retry backoff that is slower than simply pacing the
    # calls, so both concurrency and rate are capped conservatively by default.
    extraction_workers: int = Field(
        default=8,
        validation_alias=AliasChoices("EXTRACTION_WORKERS"),
        description="Concurrent LLM extraction calls. Raise on a paid model.",
    )
    extraction_requests_per_minute: int = Field(
        default=30,
        validation_alias=AliasChoices("EXTRACTION_RPM"),
        description="Request pacing. 0 disables throttling.",
    )

    # ----------------------------------------------------------- Embeddings --
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("EMBEDDING_MODEL"),
        description="sentence-transformers model run locally on CPU; no API key.",
    )

    # ------------------------------------------------------------- Reranker --
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias=AliasChoices("RERANKER_MODEL"),
        description="Cross-encoder that re-scores fused candidates. Runs locally.",
    )
    use_reranker: bool = Field(
        default=True,
        validation_alias=AliasChoices("USE_RERANKER"),
        description="Disable to skip the cross-encoder (faster, slightly worse ordering).",
    )

    # --------------------------------------------------------------- Router --
    routing_mode: str = Field(
        default="llm",
        validation_alias=AliasChoices("ROUTING_MODE"),
        description="'llm' asks the model to pick a strategy; 'heuristic' uses rules "
                    "only (no extra LLM call); 'hybrid' forces vector+graph always.",
    )
    rrf_k: int = Field(
        default=60,
        validation_alias=AliasChoices("RRF_K"),
        description="Reciprocal Rank Fusion damping constant. 60 is the value from "
                    "the original RRF paper and works well without tuning.",
    )
    rrf_vector_weight: float = Field(default=1.0, validation_alias=AliasChoices("RRF_VECTOR_WEIGHT"))
    rrf_graph_weight: float = Field(default=0.8, validation_alias=AliasChoices("RRF_GRAPH_WEIGHT"))

    # ------------------------------------------------------------- Chunking --
    # Sizes are measured in *real tokens* (tiktoken), not characters, so the
    # "~200 token child / ~1000 token parent" contract is literally enforced.
    parent_chunk_tokens: int = Field(default=1000, validation_alias=AliasChoices("PARENT_CHUNK_TOKENS"))
    parent_chunk_overlap: int = Field(default=150, validation_alias=AliasChoices("PARENT_CHUNK_OVERLAP"))
    child_chunk_tokens: int = Field(default=200, validation_alias=AliasChoices("CHILD_CHUNK_TOKENS"))
    child_chunk_overlap: int = Field(default=40, validation_alias=AliasChoices("CHILD_CHUNK_OVERLAP"))

    # ------------------------------------------------------------ Retrieval --
    vector_top_k: int = Field(
        default=12,
        validation_alias=AliasChoices("VECTOR_TOP_K"),
        description="How many *child* chunks the FAISS search returns.",
    )
    parent_top_n: int = Field(
        default=5,
        validation_alias=AliasChoices("PARENT_TOP_N"),
        description="How many *parent* blocks survive de-duplication and reach the LLM.",
    )
    graph_max_hops: int = Field(default=2, validation_alias=AliasChoices("GRAPH_MAX_HOPS"))
    graph_max_triples: int = Field(default=40, validation_alias=AliasChoices("GRAPH_MAX_TRIPLES"))

    # ---------------------------------------------------------------- Paths --
    data_dir: str = Field(default="data", validation_alias=AliasChoices("DATA_DIR"))
    graph_bootstrap: bool = Field(
        default=True,
        validation_alias=AliasChoices("GRAPH_BOOTSTRAP"),
        description="On startup, load GRAPH_BOOTSTRAP_FILE into Neo4j when the "
                    "graph is empty. Makes a fresh clone work without a manual "
                    "step. Never overwrites: it runs only at zero entities.",
    )
    graph_bootstrap_file: str = Field(
        default="emirates_nbd_graph.cypher",
        validation_alias=AliasChoices("GRAPH_BOOTSTRAP_FILE"),
        description="Cypher file used by the startup bootstrap.",
    )

    docs_dir: str = Field(
        default="Company Docs",
        validation_alias=AliasChoices("DOCS_DIR"),
        description="Directory holding source PDFs. Also the allow-list root for "
                    "the /api/v1/document endpoint that backs citation links.",
    )
    default_pdf_path: str = Field(
        default="Company Docs/strategic_report_2025_emirates_nbd.pdf",
        validation_alias=AliasChoices("DEFAULT_PDF_PATH"),
    )

    # ------------------------------------------------------------------- UI --
    api_base_url: str = Field(default="http://localhost:8000", validation_alias=AliasChoices("API_BASE_URL"))

    # ------------------------------------------------- derived path helpers --
    # These are properties rather than fields so they always stay consistent
    # with data_dir and are never accidentally overridden from the environment.

    @property
    def data_path(self) -> Path:
        """Absolute path to the data directory, created on first access."""
        p = Path(self.data_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def faiss_dir(self) -> Path:
        """Where the FAISS index of child chunks is persisted."""
        p = self.data_path / "faiss_index"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def parents_path(self) -> Path:
        """JSON file holding the parent blocks keyed by parent_id."""
        return self.data_path / "parents" / "parents.json"

    @property
    def docs_path(self) -> Path:
        """Absolute path to the source-document directory."""
        p = Path(self.docs_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def jobs_dir(self) -> Path:
        """Directory holding one JSON status file per async ingestion job."""
        p = self.data_path / "jobs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a possibly-relative path against the project root.

        Lets callers pass a relative path and get a correct absolute one
        no matter what the current working directory happens to be.
        """
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (once) and return the singleton Settings instance.

    Cached because constructing it re-reads and re-parses the .env file, and
    because every module importing ``settings`` must observe the same object.
    """
    return Settings()


# Module-level singleton — the canonical way to read configuration.
settings = get_settings()

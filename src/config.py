from pydantic import Field

# Pydantic < 2.0: BaseSettings is in pydantic
# Pydantic >= 2.0: BaseSettings moved to pydantic_settings
try:
    from pydantic import BaseSettings  # type: ignore
except ImportError:  # pragma: no cover – noqa: E722
    from pydantic_settings import BaseSettings  # type: ignore


class Settings(BaseSettings):
    """Project-wide configuration accessible via environment variables.

    Each attribute can be overridden with an env-var of the same upper-snake name.
    For example::

        export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
    """

    model_name: str = Field("uiuc-convai/CALM-8B", alias="MODEL_NAME")
    function_docs: str = Field("inference/function_docs.json", alias="FUNCTION_DOCS")
    dataset_dir: str = Field("dataset", alias="DATASET_DIR")
    output_dir: str = Field("inference/output", alias="OUTPUT_DIR")
    max_tokens: int = Field(8196, alias="MAX_TOKENS")
    tensor_parallel_size: int = Field(1, alias="TENSOR_PARALLEL_SIZE")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton settings object
settings = Settings()

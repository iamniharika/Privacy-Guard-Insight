from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppSettings:
    environment: str = "local"
    mapping_vault_path: str = "data/mapping_vault.json"


def get_settings() -> AppSettings:
    """Load local environment configuration.

    TODO: Expand strongly-typed settings for paths and runtime options.
    """
    load_dotenv()
    return AppSettings(
        environment=os.getenv("APP_ENV", "local"),
        mapping_vault_path=os.getenv("MAPPING_VAULT_PATH", "data/mapping_vault.json"),
    )

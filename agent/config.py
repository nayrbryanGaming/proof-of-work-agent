"""
Configuration loader for the Proof-of-Work Agent.
Loads environment variables and exposes a config object.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


# Find the .env file relative to this module
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(ENV_PATH)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass
class ColosseumConfig:
    """Colosseum API configuration."""
    api_key: str
    base_url: str


@dataclass
class GroqConfig:
    """Groq API configuration (EFFICIENT!)."""
    api_key: str
    model: str


@dataclass
class OpenAIConfig:
    """OpenAI API configuration (fallback)."""
    api_key: str
    model: str


@dataclass
class AgentWalletConfig:
    """AgentWallet configuration."""
    session: str


@dataclass
class SolanaConfig:
    """Solana network configuration."""
    rpc_url: str
    program_id: str


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    heartbeat_url: str
    loop_interval: int
    log_level: str


@dataclass
class Config:
    """Main configuration container."""
    colosseum: ColosseumConfig
    groq: GroqConfig
    openai: OpenAIConfig
    agent_wallet: AgentWalletConfig
    solana: SolanaConfig
    agent: AgentConfig
    base_dir: Path
    
    # Legacy attributes for backward compatibility
    colosseum_api_key: str
    groq_api_key: str
    openai_api_key: str
    agentwallet_session: str
    solana_rpc: str
    program_id: str

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables."""
        colosseum_api_key = _env("COLOSSEUM_API_KEY")
        groq_api_key = _env("GROQ_API_KEY")
        openai_api_key = _env("OPENAI_API_KEY")
        agentwallet_session = _env("AGENTWALLET_SESSION")
        solana_rpc = _env("SOLANA_RPC", "https://api.devnet.solana.com")
        program_id = _env("PROGRAM_ID")
        
        return cls(
            colosseum=ColosseumConfig(
                api_key=colosseum_api_key,
                base_url=_env("COLOSSEUM_BASE_URL", "https://agents.colosseum.com/api"),
            ),
            groq=GroqConfig(
                api_key=groq_api_key,
                model=_env("AI_MODEL", "llama-3.3-70b-versatile"),
            ),
            openai=OpenAIConfig(
                api_key=openai_api_key,
                model=_env("OPENAI_MODEL", "gpt-3.5-turbo"),
            ),
            agent_wallet=AgentWalletConfig(
                session=agentwallet_session,
            ),
            solana=SolanaConfig(
                rpc_url=solana_rpc,
                program_id=program_id,
            ),
            agent=AgentConfig(
                heartbeat_url=_env("HEARTBEAT_URL", "https://agents.colosseum.com/heartbeat.md"),
                loop_interval=int(_env("LOOP_INTERVAL_SECONDS", "1800")),
                log_level=_env("LOG_LEVEL", "INFO"),
            ),
            base_dir=BASE_DIR,
            # Legacy attributes
            colosseum_api_key=colosseum_api_key,
            groq_api_key=groq_api_key,
            openai_api_key=openai_api_key,
            agentwallet_session=agentwallet_session,
            solana_rpc=solana_rpc,
            program_id=program_id,
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        warnings = []
        
        if not self.colosseum.api_key:
            errors.append("COLOSSEUM_API_KEY is required")
        
        # AI API key - Groq preferred, OpenAI as fallback
        if not self.groq.api_key and not self.openai.api_key:
            warnings.append("No AI API key (GROQ_API_KEY or OPENAI_API_KEY) - will use fallback templates")
        
        if not self.agent_wallet.session:
            errors.append("AGENTWALLET_SESSION is required")
        
        # PROGRAM_ID is optional during development
        if not self.solana.program_id:
            warnings.append("PROGRAM_ID not set - Solana operations will be simulated")
            
        return errors

    def get_logs_dir(self) -> Path:
        """Get the logs directory path."""
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        return logs_dir

    def get_prompts_dir(self) -> Path:
        """Get the prompts directory path."""
        return self.base_dir / "prompts"

    def get_tasks_dir(self) -> Path:
        """Get the tasks directory path."""
        return self.base_dir / "tasks"
    
    def get_data_dir(self) -> Path:
        """Get the data directory path."""
        data_dir = self.base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        return data_dir


# Global config instance
config = Config.load()

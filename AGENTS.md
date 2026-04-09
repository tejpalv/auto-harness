# Agent Instructions for Göd Agent Codebase

## Commands
- **Run agent**: `python self_improving_agent.py`
- **Comparison mode**: `python self_improving_agent.py --compare`
- **Install dependencies**: `pip install openai-agents colorama python-dotenv aiohttp`

## Architecture
- **Single-file project**: All code in `self_improving_agent.py`
- **Core classes**: `SelfImprovingAgent` (main), `StandardAgent` (comparison), `AgentState` (state management), `Logger` (colored output)
- **Tool registry**: Dynamic tool management system where agent starts minimal and adds tools as needed
- **Self-improvement cycle**: Execute → Analyze (via Claude Opus 4) → Apply improvements → Retry if needed
- **APIs**: OpenAI Agents SDK for agent execution, OpenRouter API (Claude Opus 4) for self-improvement analysis

## Code Style
- **Python 3.8+** with async/await throughout
- **Type hints**: Use `typing` module (Dict, List, Optional, Any, Callable)
- **Decorators**: `@function_tool` for all tool functions, `@dataclass` for data structures
- **Imports**: Standard library first, then third-party (openai, agents, colorama, aiohttp)
- **Logging**: Use `Logger` class methods (`.info()`, `.trace()`, `.success()`, `.error()`, `.section()`, etc.) - never print() directly
- **Error handling**: Try/except with detailed error logging, graceful fallbacks to alternative APIs
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Config**: Environment variables via `.env` file (OPENAI_API_KEY, OPENROUTER_API_KEY)

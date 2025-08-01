# Göd Agent: Recursive Self-Improving Large Language Models

## Motivation

Traditional AI agents are static - they're built with a fixed set of tools, prompts, and configurations that remain unchanged throughout their operation. But what if an agent could evolve and adapt its own capabilities based on the tasks it encounters?

This project demonstrates a revolutionary approach: **an AI agent that continuously improves itself**. After each interaction, the agent analyzes its own performance and makes strategic modifications to become more effective at future tasks.

### Why Self-Improving Agents Matter

- **Adaptive Intelligence**: Instead of requiring manual reconfiguration for different tasks, the agent automatically adapts its toolkit and approach
- **Emergent Capabilities**: The agent discovers and adds new tools as needed, expanding its abilities beyond its initial design
- **Performance Optimization**: System prompts and parameters are continuously refined based on real-world usage patterns
- **Multi-Agent Orchestration**: The agent can create specialized sub-agents to handle complex tasks that require different expertise

### The Vision

Imagine an agent that starts simple but becomes increasingly sophisticated as it encounters diverse challenges. This is not just automation - it's **adaptive automation** that gets smarter with every interaction.

## Features

This project implements a self-improving AI agent that can modify its own components after each interaction:

### Core Self-Improvement Capabilities
- **Dynamic Tool Management**: Starts minimal, adds tools as needed based on task requirements
- **Automatic Prompt Optimization**: System prompt evolves to become more effective for encountered task types
- **Model Parameter Tuning**: Temperature and other settings adjust based on performance analysis
- **Agent Orchestration**: Can create and manage specialized sub-agents for complex multi-step tasks

### Advanced Features
- **Performance Tracking**: Detailed metrics and analysis for each interaction
- **Multi-Agent Analysis**: Parallel execution of specialized agents for comprehensive task handling
- **Beautiful Terminal Output**: Color-coded, timestamped logs showing the agent's evolution
- **Conversation Memory**: Maintains context across interactions while continuously improving

## How It Works

The self-improvement cycle follows this pattern:

1. **Execution**: Agent attempts to handle a user query with current capabilities
2. **Analysis**: Performance is evaluated using a separate AI model (Claude Opus 4)
3. **Improvement**: Based on analysis, the agent modifies itself:
   - Adds relevant tools from its registry
   - Updates system prompt for better task handling
   - Adjusts model parameters for optimal performance
   - Creates specialized sub-agents when beneficial
4. **Retry**: If significant improvements were made, re-runs the query with enhanced capabilities
5. **Learning**: Records improvements for future reference

This creates an agent that becomes increasingly capable and specialized for the types of tasks it encounters.

## Requirements

- Python 3.8+
- OpenAI API key (for the agents framework)
- OpenRouter API key (optional, can use OpenAI key)

## Installation

```bash
pip install openai-agents colorama python-dotenv aiohttp
```

## Setup

Create a `.env` file in the project directory:
```
OPENAI_API_KEY=your-openai-key
OPENROUTER_API_KEY=your-openrouter-key  # Optional, will use OPENAI_API_KEY if not set
```

Note: OpenRouter supports using your OpenAI API key, so you can use the same key for both.

## Usage

Run the self-improving agent:
```bash
python self_improving_agent.py
```

### Interactive Commands

- Type any prompt to interact with the agent
- `status` - Show current agent configuration and improvement history
- `quit` or `exit` - Exit the program

### Comparison Mode

To see the self-improving agent vs standard agent comparison:
```bash
python self_improving_agent.py --compare
```

## Available Tools

The agent can dynamically add these tools based on task requirements:

### Basic Tools
- `calculate`: Evaluate mathematical expressions
- `get_current_time`: Get current date and time
- `create_memory`: Store key-value pairs for later use
- `recall_memory`: Retrieve stored information
- `word_count`: Count words in text
- `reverse_text`: Reverse text strings
- `format_list`: Format comma-separated lists
- `web_search`: Search the web for information

### Advanced Agent Orchestration
- `create_and_run_agent`: Create specialized sub-agents with custom instructions and tools
- `analyze_with_multiple_agents`: Deploy multiple specialized agents in parallel for comprehensive analysis

## Agent Orchestration Examples

The agent can create specialized sub-agents for complex tasks:

### Single Sub-Agent Creation
```
User: "Use a specialized math agent to calculate compound interest"
→ Agent creates MathAgent with calculate tools and mathematical instructions
→ Delegates the calculation task to the specialized agent
→ Returns comprehensive mathematical analysis
```

### Multi-Agent Analysis
```
User: "Analyze this text with multiple perspectives"
→ Agent creates SummaryAgent, SentimentAgent, KeywordAgent in parallel
→ Each agent analyzes the text from their specialized perspective
→ Results are combined into comprehensive analysis
```

## Example Interaction

```
User: "Calculate the area of a circle with radius 5, then analyze this sentence for word count"

Agent Evolution:
1. Initial: Only has calculate and get_current_time tools
2. Analysis: Realizes it needs word_count tool for complete task handling
3. Improvement: Adds word_count tool and updates prompt
4. Retry: Successfully handles both parts of the query
5. Result: More capable agent for future mathematical + text analysis tasks
```

## Architecture

```
SelfImprovingAgent
├── AgentState (current configuration)
│   ├── System Prompt (evolves)
│   ├── Available Tools (grows dynamically)
│   ├── Model Settings (optimizes)
│   └── Improvement History (tracks changes)
├── Tool Registry (all possible tools)
├── Performance Analysis (Claude Opus 4)
└── Agent Orchestration (sub-agent creation)

Flow:
User Query → Agent Execution → Performance Analysis → Self-Improvement → Enhanced Agent
```

## The Future of Adaptive AI

This project demonstrates a fundamental shift from static to adaptive AI systems. Instead of building agents with fixed capabilities, we can create agents that evolve and specialize based on the challenges they encounter.

The implications are profound:
- **Personalized AI**: Agents that adapt to individual user needs and preferences
- **Emergent Intelligence**: Capabilities that emerge from the interaction of self-improvement and real-world tasks
- **Scalable Expertise**: Single agents that can develop specialized knowledge across multiple domains
- **Continuous Learning**: Systems that get better over time without manual intervention

This is just the beginning of what's possible when we give AI the ability to improve itself.
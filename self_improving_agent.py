#!/usr/bin/env python3
"""
Self-Improving Agent System using OpenAI Agents Framework

This system creates an agent that can modify its own components after each interaction:
- System prompt
- Available tools
- Tool descriptions and context
- Model settings
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from colorama import init, Fore, Style, Back
import inspect
from dotenv import load_dotenv

from agents import (
    Agent, Runner, RunConfig, ModelSettings, 
    function_tool, FunctionTool, WebSearchTool,
    RunResult, RunResultStreaming
)
import openai
import aiohttp

# Load environment variables from .env file
load_dotenv()

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Configure API clients
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))  # Can use OpenAI key for OpenRouter

@dataclass
class AgentState:
    """Represents the current state of the agent that can be modified"""
    system_prompt: str
    available_tools: List[Any] = field(default_factory=list)
    tool_descriptions: Dict[str, str] = field(default_factory=dict)
    model_settings: ModelSettings = field(default_factory=lambda: ModelSettings(temperature=0.7))
    context_window: List[str] = field(default_factory=list)
    improvement_history: List[Dict[str, Any]] = field(default_factory=list)

class Logger:
    """Enhanced logger for beautiful terminal output"""
    
    @staticmethod
    def header(text: str):
        print(f"\n{Back.BLUE}{Fore.WHITE}{'='*80}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{text:^80}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}{'='*80}{Style.RESET_ALL}\n")
    
    @staticmethod
    def section(text: str):
        print(f"\n{Fore.CYAN}{'─'*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}▶ {text}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─'*60}{Style.RESET_ALL}")
    
    @staticmethod
    def info(label: str, text: str):
        print(f"{Fore.GREEN}[{label}]{Style.RESET_ALL} {text}")
    
    @staticmethod
    def warning(text: str):
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def error(text: str):
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def success(text: str):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def agent_response(text: str):
        print(f"{Fore.MAGENTA}🤖 Agent:{Style.RESET_ALL} {text}")
    
    @staticmethod
    def improvement(text: str):
        print(f"{Fore.YELLOW}🔧 Improvement:{Style.RESET_ALL} {text}")
    
    @staticmethod
    def trace(operation: str, details: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{Fore.BLUE}[{timestamp}] {operation}{Style.RESET_ALL} {Fore.WHITE}{details}{Style.RESET_ALL}")
    
    @staticmethod
    def tool_call(tool_name: str, status: str = "CALLING"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if status == "CALLING":
            print(f"{Fore.YELLOW}[{timestamp}] 🔧 TOOL CALL:{Style.RESET_ALL} {Fore.CYAN}{tool_name}{Style.RESET_ALL}")
        elif status == "SUCCESS":
            print(f"{Fore.GREEN}[{timestamp}] ✓ TOOL RESULT:{Style.RESET_ALL} {Fore.CYAN}{tool_name}{Style.RESET_ALL}")
        elif status == "ERROR":
            print(f"{Fore.RED}[{timestamp}] ✗ TOOL ERROR:{Style.RESET_ALL} {Fore.CYAN}{tool_name}{Style.RESET_ALL}")
    
    @staticmethod
    def context(action: str, text: str):
        """Log context building/removal actions"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if action == "ADD":
            print(f"{Fore.BLUE}[{timestamp}] + CONTEXT:{Style.RESET_ALL} {text[:100]}...")
        elif action == "REMOVE":
            print(f"{Fore.RED}[{timestamp}] - CONTEXT:{Style.RESET_ALL} {text[:100]}...")
        elif action == "CURRENT":
            print(f"{Fore.WHITE}[{timestamp}] = CONTEXT:{Style.RESET_ALL} {text}")
    
    @staticmethod
    def sub_agent(agent_name: str, status: str):
        """Log sub-agent lifecycle events"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if status == "CREATED":
            print(f"{Fore.MAGENTA}[{timestamp}] 🤖 SUB-AGENT CREATED:{Style.RESET_ALL} {agent_name}")
        elif status == "RUNNING":
            print(f"{Fore.YELLOW}[{timestamp}] ⚡ SUB-AGENT RUNNING:{Style.RESET_ALL} {agent_name}")
        elif status == "COMPLETED":
            print(f"{Fore.GREEN}[{timestamp}] ✓ SUB-AGENT DONE:{Style.RESET_ALL} {agent_name}")
        elif status == "FAILED":
            print(f"{Fore.RED}[{timestamp}] ✗ SUB-AGENT FAILED:{Style.RESET_ALL} {agent_name}")

# Tool implementations
@function_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    Logger.tool_call("calculate", "CALLING")
    Logger.trace("  Input", expression)
    try:
        # Safe math evaluation
        import math
        safe_dict = {
            'abs': abs, 'round': round, 'pow': pow, 'sum': sum,
            'max': max, 'min': min, 'len': len,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'pi': math.pi, 'e': math.e
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        Logger.tool_call("calculate", "SUCCESS")
        Logger.trace("  Result", str(result))
        return str(result)
    except Exception as e:
        Logger.tool_call("calculate", "ERROR")
        Logger.error(f"  {str(e)}")
        return f"Error: {str(e)}"

@function_tool
def get_current_time() -> str:
    """Get the current date and time"""
    Logger.tool_call("get_current_time", "CALLING")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Logger.tool_call("get_current_time", "SUCCESS")
    Logger.trace("  Result", current_time)
    return current_time

@function_tool
def create_memory(key: str, value: str) -> str:
    """Store a memory for later retrieval"""
    Logger.tool_call("create_memory", "CALLING")
    Logger.trace("  Key", key)
    Logger.trace("  Value", value)
    # In a real implementation, this would use a persistent store
    memory_store = getattr(create_memory, '_store', {})
    memory_store[key] = value
    create_memory._store = memory_store
    Logger.tool_call("create_memory", "SUCCESS")
    Logger.context("ADD", f"Memory: {key} = {value}")
    return f"Memory stored: {key}"

@function_tool
def recall_memory(key: str) -> str:
    """Retrieve a stored memory"""
    Logger.tool_call("recall_memory", "CALLING")
    Logger.trace("  Key", key)
    memory_store = getattr(create_memory, '_store', {})
    value = memory_store.get(key, "Memory not found")
    Logger.tool_call("recall_memory", "SUCCESS")
    Logger.trace("  Result", value)
    if value != "Memory not found":
        Logger.context("CURRENT", f"Memory: {key} = {value}")
    return value

# Additional tools that can be dynamically added
@function_tool
def word_count(text: str) -> str:
    """Count the number of words in a text"""
    Logger.tool_call("word_count", "CALLING")
    Logger.trace("  Text length", f"{len(text)} characters")
    count = len(text.split())
    Logger.tool_call("word_count", "SUCCESS")
    Logger.trace("  Result", f"{count} words")
    return f"Word count: {count}"

@function_tool
def reverse_text(text: str) -> str:
    """Reverse the given text"""
    Logger.tool_call("reverse_text", "CALLING")
    Logger.trace("  Input", text[:50] + "..." if len(text) > 50 else text)
    reversed_text = text[::-1]
    Logger.tool_call("reverse_text", "SUCCESS")
    Logger.trace("  Result", reversed_text[:50] + "..." if len(reversed_text) > 50 else reversed_text)
    return reversed_text

@function_tool
def format_list(items: str) -> str:
    """Format comma-separated items as a numbered list"""
    Logger.tool_call("format_list", "CALLING")
    Logger.trace("  Input", items)
    item_list = [item.strip() for item in items.split(',')]
    formatted = "\n".join([f"{i+1}. {item}" for i, item in enumerate(item_list)])
    Logger.tool_call("format_list", "SUCCESS")
    Logger.trace("  Formatted", f"{len(item_list)} items")
    return formatted

@function_tool
async def create_and_run_agent(
    task: str,
    agent_name: str = "SubAgent",
    instructions: str = "You are a helpful AI assistant. Complete the given task.",
    tool_names: str = "calculate,get_current_time",
    temperature: float = 0.7
) -> str:
    """Create and run a specialized sub-agent with specific instructions and tools.
    
    Args:
        task: The task or query to give to the sub-agent
        agent_name: Name for the sub-agent (default: SubAgent)
        instructions: System prompt for the sub-agent
        tool_names: Comma-separated list of tool names to give the sub-agent (from available tools)
        temperature: Temperature setting for the sub-agent (0.0-1.0)
    
    Returns:
        The response from the sub-agent
    """
    try:
        Logger.section(f"Creating Sub-Agent: {agent_name}")
        Logger.info("Task", task[:100] + "..." if len(task) > 100 else task)
        Logger.info("Instructions", instructions[:100] + "..." if len(instructions) > 100 else instructions)
        Logger.info("Temperature", str(temperature))
        
        # Parse tool names and get tools from registry
        requested_tools = [name.strip() for name in tool_names.split(',')]
        available_tools = []
        
        # Get the tool registry from the global context (we'll pass it in)
        tool_registry = create_and_run_agent._tool_registry if hasattr(create_and_run_agent, '_tool_registry') else {}
        
        Logger.info("Requested Tools", ", ".join(requested_tools))
        
        for tool_name in requested_tools:
            if tool_name in tool_registry:
                available_tools.append(tool_registry[tool_name])
                Logger.success(f"✓ Tool added: {tool_name}")
            else:
                Logger.warning(f"✗ Tool not found: {tool_name}")
        
        # Create the sub-agent
        Logger.trace("Creating agent instance", f"with {len(available_tools)} tools")
        sub_agent = Agent(
            name=agent_name,
            instructions=instructions,
            tools=available_tools,
            model_settings=ModelSettings(temperature=temperature)
        )
        
        Logger.sub_agent(agent_name, "CREATED")
        
        # Run the sub-agent
        Logger.sub_agent(agent_name, "RUNNING")
        start_time = datetime.now()
        
        result = await Runner.run(sub_agent, task)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Log tool usage
        tool_calls = [item for item in result.new_items if hasattr(item, 'tool_name')]
        if tool_calls:
            Logger.info("Tools Used", f"{len(tool_calls)} tool calls made")
            for tool_call in tool_calls:
                Logger.trace(f"  → {tool_call.tool_name}", str(tool_call.input)[:100] + "..." if len(str(tool_call.input)) > 100 else str(tool_call.input))
        
        response = result.final_output
        
        Logger.sub_agent(agent_name, "COMPLETED")
        Logger.success(f"Sub-Agent completed in {execution_time:.2f}s")
        Logger.info("Response Preview", response[:200] + "..." if len(response) > 200 else response)
        
        return f"[Sub-Agent {agent_name} Response]:\n{response}"
        
    except Exception as e:
        error_msg = f"Error creating/running sub-agent: {str(e)}"
        Logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

@function_tool
async def analyze_with_multiple_agents(text: str, analysis_types: str = "summary,sentiment,keywords") -> str:
    """Analyze text using multiple specialized agents working in parallel.
    
    Args:
        text: The text to analyze
        analysis_types: Comma-separated list of analysis types (summary,sentiment,keywords,entities,topics)
    
    Returns:
        Combined analysis results from all agents
    """
    try:
        Logger.section("Multi-Agent Analysis Starting")
        Logger.info("Text Length", f"{len(text)} characters")
        Logger.info("Analysis Types", analysis_types)
        
        # Parse analysis types
        types = [t.strip() for t in analysis_types.split(',')]
        
        # Define agent configurations for different analysis types
        agent_configs = {
            'summary': {
                'name': 'SummaryAgent',
                'instructions': 'You are an expert at creating concise summaries. Summarize the given text in 2-3 sentences.',
                'tools': 'word_count'
            },
            'sentiment': {
                'name': 'SentimentAgent',
                'instructions': 'You are a sentiment analysis expert. Analyze the emotional tone and sentiment of the text. Rate it as positive, negative, or neutral with a confidence score.',
                'tools': ''
            },
            'keywords': {
                'name': 'KeywordAgent',
                'instructions': 'You are a keyword extraction expert. Extract the 5 most important keywords or phrases from the text.',
                'tools': 'word_count'
            },
            'entities': {
                'name': 'EntityAgent',
                'instructions': 'You are a named entity recognition expert. Identify all people, places, organizations, and other entities in the text.',
                'tools': ''
            },
            'topics': {
                'name': 'TopicAgent',
                'instructions': 'You are a topic modeling expert. Identify the main topics and themes discussed in the text.',
                'tools': ''
            }
        }
        
        # Create tasks for each analysis type
        tasks = []
        agent_names = []
        
        Logger.info("Creating Agents", f"Preparing {len(types)} specialized agents...")
        
        for analysis_type in types:
            if analysis_type in agent_configs:
                config = agent_configs[analysis_type]
                agent_names.append((analysis_type, config['name']))
                
                # Get the create_and_run_agent function
                create_agent_fn = analyze_with_multiple_agents._create_and_run_agent if hasattr(analyze_with_multiple_agents, '_create_and_run_agent') else create_and_run_agent
                
                Logger.trace(f"Preparing {config['name']}", f"for {analysis_type} analysis")
                
                task = create_agent_fn(
                    task=f"Analyze this text: {text}",
                    agent_name=config['name'],
                    instructions=config['instructions'],
                    tool_names=config['tools'],
                    temperature=0.3
                )
                tasks.append(task)
        
        # Run all agents in parallel
        Logger.section(f"Parallel Execution - {len(tasks)} Agents")
        Logger.info("Starting Time", datetime.now().strftime("%H:%M:%S.%f")[:-3])
        
        start_time = datetime.now()
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        Logger.success(f"All agents completed in {total_time:.2f}s (parallel)")
        
        # Combine results
        Logger.section("Combining Results")
        combined_results = []
        
        for i, (analysis_type, agent_name) in enumerate(agent_names):
            Logger.info(f"Result {i+1}", f"{agent_name} - {analysis_type}")
            result_text = results[i]
            # Extract just the response part if it has the [Sub-Agent Response]: prefix
            if "[Sub-Agent" in result_text and "Response]:" in result_text:
                result_text = result_text.split("Response]:", 1)[1].strip()
            combined_results.append(f"\n{'='*60}\n{analysis_type.upper()} ANALYSIS\n{'='*60}\n{result_text}")
        
        final_result = "\n".join(combined_results)
        Logger.success(f"Multi-agent analysis complete - {len(final_result)} characters")
        
        return final_result
        
    except Exception as e:
        error_msg = f"Error in multi-agent analysis: {str(e)}"
        Logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

class SelfImprovingAgent:
    """Main self-improving agent system"""
    
    def __init__(self):
        # Tool registry - all available tools
        self.tool_registry = {
            'calculate': calculate,
            'get_current_time': get_current_time,
            'create_memory': create_memory,
            'recall_memory': recall_memory,
            'word_count': word_count,
            'reverse_text': reverse_text,
            'format_list': format_list,
            'web_search': WebSearchTool(),
            'analyze_with_multiple_agents': analyze_with_multiple_agents
        }
        
        # Add create_and_run_agent to the registry and give it access to the tool registry
        self.tool_registry['create_and_run_agent'] = create_and_run_agent
        # Store the tool registry reference in the function so it can access other tools
        create_and_run_agent._tool_registry = self.tool_registry
        
        # Make analyze_with_multiple_agents aware of create_and_run_agent
        analyze_with_multiple_agents._create_and_run_agent = create_and_run_agent
        
        # Start with minimal tools
        self.state = AgentState(
            system_prompt="You are a helpful AI assistant. Be concise and accurate.",
            available_tools=[
                self.tool_registry['calculate'],
                self.tool_registry['get_current_time']
            ]
        )
        self.conversation_history = []
        self.improvement_count = 0
        
    async def create_agent(self) -> Agent:
        """Create an agent with the current state"""
        tools = []
        
        # Convert function tools and get their names for logging
        tool_names = []
        for tool in self.state.available_tools:
            if isinstance(tool, FunctionTool):
                tools.append(tool)
                tool_names.append(tool.name if hasattr(tool, 'name') else 'FunctionTool')
            elif hasattr(tool, '__wrapped__'):  # function_tool decorated
                # The function_tool decorator already creates a FunctionTool
                tools.append(tool)
                # Try to get the name from the wrapped function
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif hasattr(tool, '__name__'):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append('UnknownTool')
            else:
                tools.append(tool)
                tool_names.append(tool.__class__.__name__)
        
        Logger.trace("Creating agent", f"with tools: {', '.join(tool_names)}")
        
        agent = Agent(
            name="SelfImprovingAgent",
            instructions=self.state.system_prompt,
            tools=tools,
            model_settings=self.state.model_settings
        )
        
        return agent
    
    async def analyze_and_improve(self, query: str, response: str, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the interaction and suggest improvements"""
        Logger.section("Self-Improvement Analysis")
        
        # Get current tool names
        current_tool_names = []
        for tool in self.state.available_tools:
            if hasattr(tool, '__name__'):
                current_tool_names.append(tool.__name__)
            elif hasattr(tool, '__class__'):
                current_tool_names.append(tool.__class__.__name__)
        
        # Get available tools not currently in use
        available_tool_names = [name for name in self.tool_registry.keys() 
                               if name not in current_tool_names and name != 'WebSearchTool']
        
        # Create improvement prompt
        improvement_prompt = f"""
You are an AI system optimizer. Analyze this interaction and suggest improvements.

CURRENT AGENT STATE:
- System Prompt: {self.state.system_prompt}
- Available Tools: {current_tool_names}
- Model Temperature: {self.state.model_settings.temperature}

TOOLS THAT COULD BE ADDED:
{json.dumps({name: str(self.tool_registry[name].__doc__) for name in available_tool_names}, indent=2)}

Note: The 'create_and_run_agent' tool allows the agent to create and delegate tasks to specialized sub-agents.

RECENT INTERACTION:
- User Query: {query}
- Agent Response: {response}
- Performance Metrics: {json.dumps(performance_metrics, indent=2)}

CONVERSATION HISTORY:
{json.dumps(self.conversation_history[-5:], indent=2)}

Based on this analysis, suggest specific improvements to:
1. System prompt (make it more effective for the types of queries being asked)
2. Tool selection (should any tools be added/removed/modified?)
3. Model settings (temperature, etc.)
4. Any other improvements

For tools, only suggest adding tools if they would have helped with recent queries.
Consider if delegating to specialized sub-agents via 'create_and_run_agent' would improve performance.

Respond in JSON format:
{{
    "improvements": {{
        "system_prompt": "new prompt or null if no change",
        "add_tools": ["tool_names from available tools"],
        "remove_tools": ["tool_names from current tools"],
        "temperature": 0.7,
        "reasoning": "explanation of why these changes will help"
    }},
    "performance_assessment": {{
        "strengths": ["what went well"],
        "weaknesses": ["what could be better"],
        "score": 0-10
    }}
}}
"""
        
        try:
            # Use OpenRouter for improvement analysis
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/self-improving-agent",
                    "X-Title": "Self-Improving Agent"
                }
                
                data = {
                    "model": "anthropic/claude-opus-4",  # Using Claude Opus 4 via OpenRouter
                    "messages": [{"role": "user", "content": improvement_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result['choices'][0]['message']['content']
                    else:
                        # Fall back to OpenAI if OpenRouter fails
                        error_text = await response.text()
                        Logger.warning(f"OpenRouter API failed ({response.status}): {error_text}")
                        
                        from openai import AsyncOpenAI
                        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        
                        # Try o3 via OpenRouter as fallback
                        data_fallback = {
                            "model": "openai/o3",
                            "messages": [{"role": "user", "content": improvement_prompt}],
                            "temperature": 0.3,
                            "max_tokens": 1000
                        }
                        
                        async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=data_fallback
                        ) as fallback_response:
                            if fallback_response.status == 200:
                                fallback_result = await fallback_response.json()
                                response_text = fallback_result['choices'][0]['message']['content']
                            else:
                                # Last resort: use OpenAI directly
                                completion = await openai_client.chat.completions.create(
                                    model="gpt-4-turbo-preview",
                                    messages=[{"role": "user", "content": improvement_prompt}],
                                    temperature=0.3,
                                    max_tokens=1000
                                )
                                response_text = completion.choices[0].message.content
            
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                improvement_data = json.loads(json_match.group())
            else:
                Logger.warning("Could not extract JSON from improvement response")
                # Return a default improvement suggestion
                return {
                    "improvements": {
                        "system_prompt": None,
                        "add_tools": [],
                        "remove_tools": [],
                        "temperature": 0.7,
                        "reasoning": "Could not parse improvement suggestions"
                    },
                    "performance_assessment": {
                        "strengths": ["Response provided"],
                        "weaknesses": ["Analysis failed"],
                        "score": 5
                    }
                }
            
            Logger.improvement(f"Analysis complete. Score: {improvement_data['performance_assessment']['score']}/10")
            Logger.info("Reasoning", improvement_data['improvements']['reasoning'])
            
            return improvement_data
            
        except Exception as e:
            Logger.error(f"Improvement analysis failed: {str(e)}")
            # Return a default improvement suggestion
            return {
                "improvements": {
                    "system_prompt": None,
                    "add_tools": [],
                    "remove_tools": [],
                    "temperature": 0.7,
                    "reasoning": f"Analysis failed: {str(e)}"
                },
                "performance_assessment": {
                    "strengths": ["Response provided"],
                    "weaknesses": ["Analysis failed"],
                    "score": 5
                }
            }
    
    async def apply_improvements(self, improvements: Dict[str, Any]):
        """Apply the suggested improvements to the agent state"""
        Logger.section("Applying Improvements")
        
        changes_made = []
        
        # Update system prompt
        if improvements['improvements']['system_prompt']:
            old_prompt = self.state.system_prompt
            new_prompt = improvements['improvements']['system_prompt']
            self.state.system_prompt = new_prompt
            changes_made.append(f"Updated system prompt")
            Logger.success("System prompt updated")
            
            # Show the diff instead of full prompts
            import difflib
            diff = difflib.unified_diff(
                old_prompt.splitlines(keepends=True),
                new_prompt.splitlines(keepends=True),
                fromfile='old_prompt',
                tofile='new_prompt',
                n=3
            )
            diff_text = ''.join(diff)
            if diff_text:
                Logger.info("Prompt Diff", "")
                for line in diff_text.splitlines():
                    if line.startswith('+') and not line.startswith('+++'):
                        print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
                    elif line.startswith('-') and not line.startswith('---'):
                        print(f"{Fore.RED}{line}{Style.RESET_ALL}")
                    elif line.startswith('@'):
                        print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.WHITE}{line}{Style.RESET_ALL}")
        
        # Add tools
        if improvements['improvements'].get('add_tools'):
            for tool_name in improvements['improvements']['add_tools']:
                if tool_name in self.tool_registry:
                    tool = self.tool_registry[tool_name]
                    if tool not in self.state.available_tools:
                        self.state.available_tools.append(tool)
                        changes_made.append(f"Added tool: {tool_name}")
                        Logger.success(f"Added tool: {tool_name}")
        
        # Remove tools
        if improvements['improvements'].get('remove_tools'):
            for tool_name in improvements['improvements']['remove_tools']:
                # Find and remove the tool
                for i, tool in enumerate(self.state.available_tools):
                    if hasattr(tool, '__name__') and tool.__name__ == tool_name:
                        self.state.available_tools.pop(i)
                        changes_made.append(f"Removed tool: {tool_name}")
                        Logger.success(f"Removed tool: {tool_name}")
                        break
        
        # Update temperature
        if 'temperature' in improvements['improvements']:
            old_temp = self.state.model_settings.temperature
            new_temp = improvements['improvements']['temperature']
            self.state.model_settings.temperature = new_temp
            changes_made.append(f"Temperature changed to {new_temp}")
            Logger.success(f"Temperature updated: {old_temp} → {new_temp}")
        
        # Record improvement
        self.state.improvement_history.append({
            "timestamp": datetime.now().isoformat(),
            "improvements": improvements,
            "changes_made": changes_made
        })
        
        self.improvement_count += 1
        Logger.info("Total Improvements", str(self.improvement_count))
        
        return changes_made
    
    async def run_with_improvement(self, query: str, max_retries: int = 3) -> str:
        """Run the agent and apply self-improvements, retrying if improvements are made"""
        Logger.header(f"Self-Improving Agent - Query #{len(self.conversation_history) + 1}")
        Logger.info("User Query", query)
        
        best_response = None
        best_score = 0
        retry_count = 0
        
        while retry_count < max_retries:
            # Create agent with current state
            agent = await self.create_agent()
            
            # Run the agent
            if retry_count > 0:
                Logger.section(f"Agent Re-execution (Attempt {retry_count + 1})")
            else:
                Logger.section("Agent Execution")
            
            # Log current agent configuration
            current_tool_names = []
            for tool in self.state.available_tools:
                if hasattr(tool, '__name__'):
                    current_tool_names.append(tool.__name__)
                elif hasattr(tool, '__class__'):
                    current_tool_names.append(tool.__class__.__name__)
                elif hasattr(tool, 'name'):
                    current_tool_names.append(tool.name)
                else:
                    current_tool_names.append(str(type(tool).__name__))
            
            Logger.info("Current Tools", ", ".join(current_tool_names))
            Logger.info("Temperature", str(self.state.model_settings.temperature))
            
            start_time = datetime.now()
            
            try:
                # Run with detailed logging
                Logger.trace("Starting agent execution", "")
                result = await Runner.run(agent, query)
                response = result.final_output
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Debug: Log what types of items we're getting
                Logger.trace("New items", f"{len(result.new_items)} items")
                for idx, item in enumerate(result.new_items):
                    item_type = type(item).__name__
                    item_attrs = [attr for attr in dir(item) if not attr.startswith('_')]
                    Logger.trace(f"  Item {idx}", f"Type: {item_type}, Attrs: {item_attrs[:5]}...")
                
                # Log all tool calls made during execution
                tool_calls = []
                tool_responses = []
                
                for item in result.new_items:
                    item_type = type(item).__name__
                    # Check for tool call items by type name
                    if item_type == 'ToolCallItem':
                        tool_calls.append(item)
                    elif item_type == 'ToolCallOutputItem':
                        tool_responses.append(item)
                
                # Count tools used for metrics
                tools_used_count = len(tool_calls)
                
                if tool_calls or tool_responses:
                    Logger.section("Tool Calls Made")
                    
                    # Log tool calls
                    for i, tool_call in enumerate(tool_calls, 1):
                        # Try to get tool name from various possible attributes
                        tool_name = "Unknown"
                        if hasattr(tool_call, 'tool_name'):
                            tool_name = tool_call.tool_name
                        elif hasattr(tool_call, 'name'):
                            tool_name = tool_call.name
                        elif hasattr(tool_call, 'raw_item'):
                            # Check raw_item attributes
                            raw_item = tool_call.raw_item
                            if hasattr(raw_item, 'id'):
                                tool_name = raw_item.id
                            elif hasattr(raw_item, 'function'):
                                if isinstance(raw_item.function, dict):
                                    tool_name = raw_item.function.get('name', 'Unknown')
                                elif hasattr(raw_item.function, 'name'):
                                    tool_name = raw_item.function.name
                            elif hasattr(raw_item, 'name'):
                                tool_name = raw_item.name
                        
                        Logger.info(f"Tool Call {i}", tool_name)
                        
                        # Try to get arguments
                        if hasattr(tool_call, 'arguments'):
                            Logger.trace("  Arguments", str(tool_call.arguments)[:200])
                        elif hasattr(tool_call, 'raw_item'):
                            raw_item = tool_call.raw_item
                            if hasattr(raw_item, 'function') and isinstance(raw_item.function, dict):
                                args = raw_item.function.get('arguments', {})
                                Logger.trace("  Arguments", str(args)[:200])
                            elif hasattr(raw_item, 'arguments'):
                                Logger.trace("  Arguments", str(raw_item.arguments)[:200])
                    
                    # Log tool responses
                    for i, tool_response in enumerate(tool_responses, 1):
                        Logger.info(f"Tool Response {i}", "Result")
                        if hasattr(tool_response, 'output'):
                            Logger.trace("  Output", str(tool_response.output)[:200])
                        elif hasattr(tool_response, 'content'):
                            Logger.trace("  Output", str(tool_response.content)[:200])
                else:
                    Logger.info("Tool Calls", "No tools were used")
                
                # Log response
                Logger.section("Agent Response")
                Logger.agent_response(response)
                Logger.trace("Execution time", f"{execution_time:.2f}s")
                
                # Collect performance metrics
                performance_metrics = {
                    "execution_time": execution_time,
                    "response_length": len(response),
                    "tools_used": tools_used_count,
                    "retry_count": retry_count
                }
                
                Logger.trace("Performance Metrics", json.dumps(performance_metrics, indent=2))
                
                # Analyze and improve
                improvements = await self.analyze_and_improve(query, response, performance_metrics)
                
                current_score = improvements['performance_assessment']['score'] if improvements else 10
                
                # Keep track of best response
                if current_score > best_score:
                    best_response = response
                    best_score = current_score
                
                # Check if we should improve and retry
                if improvements and current_score < 8:
                    changes_made = await self.apply_improvements(improvements)
                    
                    # If significant improvements were made, retry
                    if any('tool' in change for change in changes_made) or 'system prompt' in str(changes_made):
                        retry_count += 1
                        Logger.info("Retry", f"Re-running query with improvements (Score: {current_score}/10)")
                        continue
                    else:
                        Logger.success("Minor improvements applied. Using current response.")
                        break
                else:
                    Logger.success(f"Performance is optimal. Score: {current_score}/10")
                    break
                    
            except Exception as e:
                Logger.error(f"Agent execution failed: {str(e)}")
                import traceback
                traceback.print_exc()
                if best_response:
                    Logger.warning("Returning best response from previous attempts")
                    return best_response
                raise
        
        # Store final conversation in history
        self.conversation_history.append({
            "query": query,
            "response": best_response,
            "final_score": best_score,
            "total_attempts": retry_count + 1,
            "timestamp": datetime.now().isoformat()
        })
        
        if retry_count > 0:
            Logger.success(f"Completed after {retry_count + 1} attempts. Final score: {best_score}/10")
        
        return best_response

class StandardAgent:
    """Standard agent for comparison"""
    
    def __init__(self):
        self.agent = Agent(
            name="StandardAgent",
            instructions="You are a helpful AI assistant. Be concise and accurate.",
            tools=[
                calculate,
                get_current_time,
                create_memory,
                recall_memory,
                WebSearchTool()
            ],
            model_settings=ModelSettings(temperature=0.7)
        )
    
    async def run(self, query: str) -> str:
        """Run the standard agent"""
        Logger.info("Standard Agent Query", query)
        result = await Runner.run(self.agent, query)
        response = result.final_output
        Logger.agent_response(response)
        return response

async def compare_agents():
    """Compare self-improving agent vs standard agent"""
    Logger.header("Agent Comparison Test")
    
    # Test prompts designed to show improvement
    test_prompts = [
        "Calculate the compound interest on $1000 at 5% annual rate for 3 years",
        "Count the words in this sentence: The quick brown fox jumps over the lazy dog",
        "Create a memory called 'username' with value 'Alice' and then recall it",
        "Format this list: apple, banana, cherry, date, elderberry"
    ]
    
    # Initialize agents
    self_improving = SelfImprovingAgent()
    standard = StandardAgent()
    
    Logger.section("Testing Self-Improving Agent")
    si_responses = []
    for i, prompt in enumerate(test_prompts):
        Logger.info(f"Test {i+1}/4", prompt)
        response = await self_improving.run_with_improvement(prompt)
        si_responses.append(response)
        print()  # Spacing
    
    Logger.section("Testing Standard Agent")
    std_responses = []
    for i, prompt in enumerate(test_prompts):
        Logger.info(f"Test {i+1}/4", prompt)
        response = await standard.run(prompt)
        std_responses.append(response)
        print()  # Spacing
    
    # Show improvement history
    Logger.header("Self-Improvement Summary")
    Logger.info("Total Improvements Made", str(self_improving.improvement_count))
    
    for i, improvement in enumerate(self_improving.state.improvement_history):
        Logger.section(f"Improvement #{i+1}")
        Logger.info("Timestamp", improvement['timestamp'])
        Logger.info("Changes", ", ".join(improvement['changes_made']))
    
    Logger.success("Comparison test completed!")

async def interactive_session():
    """Run an interactive session with the self-improving agent"""
    from agents import set_default_openai_key
    
    # Set the OpenAI key for the agents framework
    set_default_openai_key(os.getenv("OPENAI_API_KEY"))
    
    # Create agent
    agent = SelfImprovingAgent()
    
    Logger.header("Interactive Self-Improving Agent")
    print("Type 'quit' or 'exit' to stop, 'status' to see current state")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Fore.GREEN}Enter your prompt:{Style.RESET_ALL} ").strip()
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                Logger.info("Goodbye", "Thanks for using the self-improving agent!")
                break
            
            if user_input.lower() == 'status':
                # Show current agent status
                Logger.section("Current Agent Status")
                
                current_tools = []
                for tool in agent.state.available_tools:
                    if hasattr(tool, '__name__'):
                        current_tools.append(tool.__name__)
                    elif hasattr(tool, '__class__'):
                        current_tools.append(tool.__class__.__name__)
                
                Logger.info("System Prompt", agent.state.system_prompt[:80] + "..." if len(agent.state.system_prompt) > 80 else agent.state.system_prompt)
                Logger.info("Available Tools", ", ".join(current_tools))
                Logger.info("Temperature", str(agent.state.model_settings.temperature))
                Logger.info("Total Improvements", str(agent.improvement_count))
                Logger.info("Queries Processed", str(len(agent.conversation_history)))
                continue
            
            if not user_input:
                continue
            
            # Process the query
            response = await agent.run_with_improvement(user_input)
            
        except KeyboardInterrupt:
            print("\n")
            Logger.info("Interrupted", "Use 'quit' to exit properly")
        except Exception as e:
            Logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

async def main():
    """Main entry point"""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        Logger.error("Please set OPENAI_API_KEY environment variable")
        print("\nTo set your API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return
    
    # Check if we should run comparison or interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        await compare_agents()
    else:
        await interactive_session()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
# göd-agent

A self-improving AI agent in 169 lines. One tool: bash. One loop. It grows by writing Python mods that hot-reload into its own runtime.

## How it works

The agent is a while loop: user input → LLM → tool calls → repeat. The only tool is `bash`, which can do anything — run code, edit files, curl APIs, install packages.

The trick: `~/.god-agent/mods/` contains Python files that are `exec()`'d into the agent's global namespace every turn. The agent can write mods via bash that replace its own functions, add tools, or change its model — mid-session, no restart.

```
you:   "add streaming"
agent: writes ~/.god-agent/mods/01_streaming.py (replaces chat() with streaming version)
       → mod is exec'd into globals after the bash command
       → streaming works on the same turn
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OpenRouter key to .env
python self_improving_agent.py
```

## What mods can do

A mod is a `.py` file in `~/.god-agent/mods/` that gets exec'd into the agent's globals. It can:

- Replace `chat()` (e.g. add streaming, change output format)
- Replace `run_bash()` (e.g. add Docker sandboxing)
- Mutate the `tools` list (e.g. add a web_search tool)
- Reassign `MODEL` (e.g. switch to a cheaper model)
- Reassign `SYSTEM` (e.g. add persistent instructions)
- Anything — it's just Python in the same namespace

Mods are sorted by filename (`01_`, `02_`, ...) and tracked by mtime so only new/changed files are re-exec'd.

## Architecture

```
main()                          ← unkillable supervisor, defined last, never replaced
  reload_mods()                 ← exec new/changed .py files from ~/.god-agent/mods/
  globals()["chat"](messages)   ← dispatch to latest chat(), even if a mod just replaced it

chat()                          ← replaceable by mods
  while True:
    LLM call (MODEL, tools)
    if text response → print, return
    if tool calls → run bash, append results
      reload_mods()             ← check if bash just wrote a new mod
      if globals()["chat"] is not chat → hand off to new version
```

Two reload points: before every user turn (in `main()`), and after every batch of tool calls (in `chat()`). Both dispatch through `globals()` so replacements take effect immediately.

## Files

```
self_improving_agent.py    ← the entire agent (169 lines, git-tracked, read-only)
~/.god-agent/mods/         ← user's accumulated growth (persists across sessions)
```

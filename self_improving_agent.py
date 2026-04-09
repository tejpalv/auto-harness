#!/usr/bin/env python3
"""Self-improving agent. One tool: bash. One loop. It can edit this file."""

import os, sys, json, subprocess, threading, time, itertools
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "anthropic/claude-opus-4-6"
SELF = os.path.abspath(__file__)
GROWTH_DIR = os.path.expanduser("~/.god-agent")

# ── Growth directory ────────────────────────────────────────────────────
os.makedirs(GROWTH_DIR, exist_ok=True)

SYSTEM = f"""You are an AI agent with one tool: bash. You can do anything.
Your source: {SELF} (read-only, git-tracked)
Your growth: {GROWTH_DIR}/ — .py files here are exec'd into your runtime every turn. Read your source to understand what you can replace."""

tools = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

# ── Spinner ───────────────────────────────────────────────────────────────
class Spinner:
    _F = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    def __init__(self, msg="thinking"): self._msg, self._stop = msg, threading.Event()
    def __enter__(self):
        def spin():
            for f in itertools.cycle(self._F):
                if self._stop.is_set(): break
                sys.stdout.write(f"\r\033[36m{f} {self._msg}\033[0m"); sys.stdout.flush(); time.sleep(0.08)
            sys.stdout.write(f"\r{' ' * (len(self._msg) + 4)}\r"); sys.stdout.flush()
        self._t = threading.Thread(target=spin, daemon=True); self._t.start(); return self
    def __exit__(self, *a): self._stop.set(); self._t.join()


# ── Core functions ───────────────────────────────────────────────────────


def run_bash(command: str) -> str:
    """Execute a bash command and return combined output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        return output[:10000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 120s"
    except Exception as e:
        return f"ERROR: {e}"


def chat(messages: list) -> None:
    """One turn: call LLM, handle tool calls or print response. Replaceable by mods."""
    while True:
        with Spinner("thinking"):
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools
            )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"\n\033[1m{msg.content}\033[0m")
            return

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\033[90m$ {args['command']}\033[0m")
            with Spinner("running"):
                result = run_bash(args["command"])
            print(result[:500] + ("..." if len(result) > 500 else ""))
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

        # Reload after tool calls — a bash command may have written a new mod
        reload_mods()
        messages[0] = {"role": "system", "content": SYSTEM}
        # If a mod just replaced chat(), hand off to the new version
        if globals()["chat"] is not chat:
            globals()["chat"](messages)
            return


# ── Mod loader: hot-reloads ~/.god-agent/mods/*.py on every chat turn ──
_mods_dir = os.path.join(GROWTH_DIR, "mods")
os.makedirs(_mods_dir, exist_ok=True)
_mod_mtimes = {}  # tracks {filepath: last_mtime} so we only re-exec changed files


def reload_mods():
    """Exec new or modified mods into globals. Called every chat turn."""
    for name in sorted(os.listdir(_mods_dir)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(_mods_dir, name)
        mtime = os.path.getmtime(path)
        if _mod_mtimes.get(path) == mtime:
            continue  # unchanged, skip
        is_new = path not in _mod_mtimes
        try:
            with open(path) as f:
                exec(f.read(), globals())
            _mod_mtimes[path] = mtime
            print(f"\033[36m  mod {'loaded' if is_new else 'reloaded'}: {name}\033[0m")
        except Exception as e:
            print(f"\033[31m  mod failed: {name}: {e}\033[0m")


# Load any existing mods on startup
reload_mods()


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY in .env or environment.")
        sys.exit(1)

    messages = [{"role": "system", "content": SYSTEM}]
    print(f"göd-agent ({MODEL})")
    print(f"source: {SELF}")
    print(f"growth: {GROWTH_DIR}/")
    print("ctrl-c to quit\n")

    while True:
        try:
            user_input = input("\033[32m> \033[0m").strip()
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            # Reload mods + system prompt BEFORE dispatching — main() owns this,
            # so even if a mod replaced chat(), this always runs
            reload_mods()
            messages[0] = {"role": "system", "content": SYSTEM}
            # Dispatch through globals() so we always call the latest chat()
            globals()["chat"](messages)
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break


if __name__ == "__main__":
    main()

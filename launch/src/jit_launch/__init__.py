"""Coding agents on a JITEngine engine: `jit-launch {pi,claude,opencode,codex}` reads the served
model from a running engine, points the agent at it for one session and starts the agent, as
`ollama launch` and `unsloth start` do.

Modules: cli (discovery, the command line and running the agent), agents (each agent's session
configuration and command).
"""

__version__ = "0.1.0a1"

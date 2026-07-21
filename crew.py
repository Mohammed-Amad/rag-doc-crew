"""
Builds the CrewAI Agents/Tasks/Crew from the jsonc config files, resolving
"custom:<name>" tool references to actual tool instances.
"""

import json
import re
import os

from crewai import Agent, Task, Crew, Process, LLM

from tools.rag_tool import DocumentRAGTool

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


CUSTOM_TOOL_REGISTRY = {
    "custom:rag_search": DocumentRAGTool(),
}


def load_jsonc(path: str) -> dict:
    """Minimal JSONC loader: strips // line comments, then json.loads."""
    with open(path, "r") as f:
        raw = f.read()
    no_comments = re.sub(r"(?m)^\s*//.*$", "", raw)
    return json.loads(no_comments)


def resolve_tools(tool_names):
    return [CUSTOM_TOOL_REGISTRY[name] for name in tool_names if name in CUSTOM_TOOL_REGISTRY]


def build_crew(question: str, groq_model: str = "llama-3.3-70b-versatile") -> Crew:
    agents_cfg = load_jsonc(os.path.join(CONFIG_DIR, "agents.jsonc"))
    tasks_cfg = load_jsonc(os.path.join(CONFIG_DIR, "tasks.jsonc"))

    llm = LLM(
        model=groq_model,
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
    )

    agents = {}
    for key, cfg in agents_cfg.items():
        agents[key] = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            tools=resolve_tools(cfg.get("tools", [])),
            allow_delegation=cfg.get("allow_delegation", False),
            verbose=cfg.get("verbose", True),
            llm=llm,
            max_iter=8,
        )

    tasks = {}
    ordered_task_keys = ["research_task", "fact_check_task", "report_task"]
    for key in ordered_task_keys:
        cfg = tasks_cfg[key]
        context_tasks = [tasks[c] for c in cfg.get("context", [])]
        tasks[key] = Task(
            description=cfg["description"].format(question=question),
            expected_output=cfg["expected_output"],
            agent=agents[cfg["agent"]],
            context=context_tasks if context_tasks else None,
        )

    crew = Crew(
        agents=list(agents.values()),
        tasks=[tasks[k] for k in ordered_task_keys],
        process=Process.sequential,
        verbose=True,
    )
    return crew

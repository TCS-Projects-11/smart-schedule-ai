"""Shared LLM setup for the scheduling assistant."""

import json
import os
from typing import Optional

from langchain.chat_models import init_chat_model

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import truststore

    # Use the Windows/OS certificate store so corporate TLS inspection
    # (custom root CA) is trusted when calling Gemini/OpenAI.
    truststore.inject_into_ssl()
except ImportError:
    pass

DEFAULT_MODEL = os.environ.get("SCHEDULING_LLM_MODEL", "google_genai:gemini-2.5-flash")

SYSTEM_INSTRUCTIONS = """You are a factory scheduling assistant.

You will be given two JSON documents:
1. SCHEDULER_INPUT - the original data given to the OR-Tools scheduler
   (workers, machines, orders, constraints).
2. SCHEDULER_OUTPUT - everything the OR-Tools solver produced (scheduled
   orders with machine/worker/start/end/lateness, unscheduled orders with
   reasons, machine utilization including busy and idle intervals, and
   summary statistics).

STRICT RULES:
- Answer ONLY using facts present in SCHEDULER_INPUT and SCHEDULER_OUTPUT.
- Never invent, guess, or assume any order, machine, worker, time, or
  statistic that is not explicitly present in the provided JSON.
- If a question cannot be answered from the provided data, respond exactly
  with: "That information is not available in the provided data." - then,
  if helpful, briefly say what data would be needed to answer it.
- When you state a time, machine, worker, or order ID, it must be an exact
  value copied from the JSON, not a paraphrase or estimate.
- For availability questions (e.g. whether a machine is free between two
  times), use that machine's available window, busy_intervals, and
  idle_intervals. A machine is free in a window only if the entire window
  sits inside its availability and does not overlap any busy interval.
- Be concise and factual. Do not add scheduling opinions or recommendations
  unless the user explicitly asks for them.
"""


def create_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Create a chat model via LangChain's provider-agnostic initializer.

    Pass any supported identifier, for example:
      - google_genai:gemini-2.5-flash
      - openai:gpt-4o-mini
    Override with SCHEDULING_LLM_MODEL in the environment.
    """
    model_id = model or DEFAULT_MODEL
    kwargs = {"temperature": temperature}

    # Last-resort override for locked-down networks. Prefer the OS trust
    # store (truststore above). Set SCHEDULING_SSL_VERIFY=false only if
    # that still fails.
    verify = os.environ.get("SCHEDULING_SSL_VERIFY", "true").strip().lower()
    if verify in ("0", "false", "no"):
        kwargs["client_args"] = {"verify": False}

    return init_chat_model(model_id, **kwargs)


def build_context_json(scheduler_input: dict, scheduler_output: dict) -> str:
    return json.dumps(
        {
            "SCHEDULER_INPUT": scheduler_input,
            "SCHEDULER_OUTPUT": scheduler_output,
        },
        indent=2,
        default=str,
    )

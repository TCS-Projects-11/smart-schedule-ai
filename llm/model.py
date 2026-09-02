"""Shared LLM setup for the scheduling assistant."""

import json
from typing import Optional
import httpx

from langchain_openai import ChatOpenAI 
client = httpx.Client(verify=False) 

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


def create_llm(temperature: float = 0.0):
    
    llm = ChatOpenAI( 
      base_url="https://genailab.tcs.in",
      model = "azure/genailab-maas-gpt-4o-mini", 
      api_key="sk-nRMnsJrM3BVTaAWCDjeVUg",
      http_client = client,
      temperature=temperature
    ) 

    return llm


def build_context_json(scheduler_input: dict, scheduler_output: dict) -> str:
    return json.dumps(
        {
            "SCHEDULER_INPUT": scheduler_input,
            "SCHEDULER_OUTPUT": scheduler_output,
        },
        indent=2,
        default=str,
    )

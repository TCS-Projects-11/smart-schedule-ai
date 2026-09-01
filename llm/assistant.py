"""
llm/assistant.py
-----------------
LLM analysis layer for the OR-Tools factory scheduler.

This module NEVER touches scheduling logic. It only ever reads two
JSON-serializable dicts that are handed to it from the outside:

    scheduler_input  -> the original factory_data.json (workers, machines,
                         orders, constraints)
    scheduler_output -> everything OR-Tools produced (scheduled orders,
                         unscheduled orders + reasons, utilization, stats)

Both are serialized to JSON and injected into the LLM's context. The LLM
is instructed to answer ONLY from that context - never from its own
general knowledge - and to say explicitly when something isn't in the
data. This keeps the module a pure "read-only reporting layer" on top of
a scheduler that has already run.

Two public capabilities:
    1. generate_summary()          -> automatic natural-language summary
    2. answer_question(question)   -> grounded Q&A over the same data

Uses LangChain + langchain-openai (ChatOpenAI). Requires OPENAI_API_KEY
to be set in the environment.
"""

import json
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "sk-nRMnsJrM3BVTaAWCDjeVUg"

# ---------------------------------------------------------------------------
# Grounding rules. This is the single most important prompt in the module:
# it's what stops the LLM from hallucinating orders, machines, or numbers
# that aren't actually in the scheduler's output.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """You are a factory scheduling assistant.

You will be given two JSON documents:
1. SCHEDULER_INPUT - the original data given to the OR-Tools scheduler
   (workers, machines, orders, constraints).
2. SCHEDULER_OUTPUT - everything the OR-Tools solver produced (scheduled
   orders with machine/worker/start/end/lateness, unscheduled orders with
   reasons, machine utilization, and summary statistics).

STRICT RULES:
- Answer ONLY using facts present in SCHEDULER_INPUT and SCHEDULER_OUTPUT.
- Never invent, guess, or assume any order, machine, worker, time, or
  statistic that is not explicitly present in the provided JSON.
- If a question cannot be answered from the provided data, respond exactly
  with: "That information is not available in the provided data." - then,
  if helpful, briefly say what data would be needed to answer it.
- When you state a time, machine, worker, or order ID, it must be an exact
  value copied from the JSON, not a paraphrase or estimate.
- Be concise and factual. Do not add scheduling opinions or recommendations
  unless the user explicitly asks for them.
"""


class SchedulingAssistant:
    """Thin, stateless wrapper around an LLM, grounded on one schedule run."""

    def __init__(
        self,
        scheduler_input: dict,
        scheduler_output: dict,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.scheduler_input = scheduler_input
        self.scheduler_output = scheduler_output

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Export it before running the assistant."
            )

        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # Pre-serialize the context once - reused for every summary/question
        # so we're not re-encoding JSON on every call.
        self._context_json = json.dumps(
            {
                "SCHEDULER_INPUT": self.scheduler_input,
                "SCHEDULER_OUTPUT": self.scheduler_output,
            },
            indent=2,
            default=str,  # safety net for any stray non-JSON-native types
        )

    # -----------------------------------------------------------------
    # Internal helper: builds the two messages every call sends -
    # (system rules) + (system-role context block) - so summary and Q&A
    # share identical grounding.
    # -----------------------------------------------------------------
    def _base_messages(self):
        return [
            SystemMessage(content=SYSTEM_INSTRUCTIONS),
            SystemMessage(
                content=f"Here is the scheduling data to use:\n\n{self._context_json}"
            ),
        ]

    # -----------------------------------------------------------------
    # Capability 1: Automatic summary
    # -----------------------------------------------------------------
    def generate_summary(self) -> str:
        """Produce a concise natural-language summary of the schedule run."""
        instruction = HumanMessage(
            content=(
                "Write a concise natural-language summary of this scheduling run. "
                "Cover, in short paragraphs or bullet points:\n"
                "- How many orders were scheduled vs. unscheduled (give counts)\n"
                "- Which specific orders are late, and which finished on time\n"
                "- Notable machine utilization (e.g. any machine that's fully "
                "  booked or mostly idle)\n"
                "- Notable worker workload (e.g. any worker with much more/less "
                "  work than others)\n"
                "- The reason for each unscheduled order\n"
                "- Any other notable scheduling issues visible in the data\n"
                "Only use facts from the provided JSON."
            )
        )
        messages = self._base_messages() + [instruction]
        response = self.llm.invoke(messages)
        return response.content

    # -----------------------------------------------------------------
    # Capability 2: Grounded question answering
    # -----------------------------------------------------------------
    def answer_question(self, question: str) -> str:
        """Answer a single natural-language question about the schedule."""
        instruction = HumanMessage(
            content=(
                f"Question: {question}\n\n"
                "Answer using only the provided SCHEDULER_INPUT and "
                "SCHEDULER_OUTPUT data. If the answer requires information "
                "not present in that data, say so explicitly."
            )
        )
        messages = self._base_messages() + [instruction]
        response = self.llm.invoke(messages)
        return response.content


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Command-line interface
#
# Usage:
#   python llm/assistant.py <scheduler_input.json> <scheduler_output.json>
#
# `scheduler_output.json` is expected to already exist - produced by a small
# adapter around the existing OR-Tools scheduler (see note in the project
# README / accompanying explanation; the core solving logic in
# factory_scheduler.py is not modified to create it).
# ---------------------------------------------------------------------------
def run_cli(input_path: str, output_path: str, model: Optional[str] = None):
    scheduler_input = load_json(input_path)
    scheduler_output = load_json(output_path)

    assistant = SchedulingAssistant(
        scheduler_input=scheduler_input,
        scheduler_output=scheduler_output,
        model=model or os.environ.get("SCHEDULING_LLM_MODEL", "gpt-4o-mini"),
    )

    print("=" * 70)
    print("AUTOMATIC SCHEDULE SUMMARY")
    print("=" * 70)
    print(assistant.generate_summary())

    print()
    print("=" * 70)
    print("ASK QUESTIONS ABOUT THE SCHEDULE (type 'exit' or 'quit' to stop)")
    print("=" * 70)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        answer = assistant.answer_question(question)
        print(answer)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python llm/assistant.py <scheduler_input.json> <scheduler_output.json>")
        sys.exit(1)

    run_cli(sys.argv[1], sys.argv[2])
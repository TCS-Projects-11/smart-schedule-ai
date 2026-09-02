"""Orchestrates summary and Q&A over one in-memory schedule run.

Does not read or write scheduler_output files. The caller (scheduler.py)
passes the factory input and the solved schedule as dicts.
"""

from typing import Optional

from llm.model import build_context_json, create_llm
from llm.qa import answer_question
from llm.summary import generate_summary


class SchedulingAssistant:
    """Thin wrapper: one LLM, one schedule context, two capabilities."""

    def __init__(
        self,
        scheduler_input: dict,
        scheduler_output: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.scheduler_input = scheduler_input
        self.scheduler_output = scheduler_output
        self.llm = create_llm(model=model, temperature=temperature)
        self._context_json = build_context_json(scheduler_input, scheduler_output)

    def generate_summary(self) -> str:
        return generate_summary(self.llm, self._context_json)

    def answer_question(self, question: str) -> str:
        return answer_question(self.llm, self._context_json, question)

    def interactive_qa(self) -> None:
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

            print(self.answer_question(question))

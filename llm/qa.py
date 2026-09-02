"""Grounded Q&A over a completed schedule run."""

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model import SYSTEM_INSTRUCTIONS


def answer_question(llm, context_json: str, question: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        SystemMessage(content=f"Here is the scheduling data to use:\n\n{context_json}"),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                "Answer using only the provided SCHEDULER_INPUT and"
                "SCHEDULER OUTPUT data. If the answer requires information"
                "not present in that data, say so explicitly."
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content

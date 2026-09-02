"""Natural-language summary of a completed schedule run."""

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model import SYSTEM_INSTRUCTIONS

SUMMARY_INSTRUCTION = (
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


def generate_summary(llm, context_json: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        SystemMessage(content=f"Here is the scheduling data to use:\n\n{context_json}"),
        HumanMessage(content=SUMMARY_INSTRUCTION),
    ]
    response = llm.invoke(messages)
    return response.content

"""
llm/reschedule_agent.py
-------------------------
A ReAct agent that can modify factory availability data based on a
natural-language change request, ask the user to CONFIRM the parsed
values before applying anything, then write the updated data to a JSON
file and re-run the existing (unmodified) OR-Tools scheduler on it.

Example: "Machine 2 will not be available between 10am and 1pm."
"""

import copy
import json
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

from llm.model import create_llm
from scheduler import create_schedule, to_hour_offset, hour_offset_to_clock, get_windows


UPDATED_DATA_PATH = Path(__file__).parent.parent / "data" / "factory_data_updated.json"

AGENT_SYSTEM_PROMPT = """You are a factory scheduling assistant that can modify machine or
worker availability, manually pin an order's start time, and re-run the
scheduler.

You have two tools:

1. update_availability_and_reschedule - for machine/worker downtime
   (e.g. "Machine 2 is down from 10am to 1pm", "Worker B is out sick
   Wednesday afternoon").

2. reschedule_order_time - for moving a SPECIFIC order to a specific
   start time (e.g. "reschedule order 1 to 11:00", "move O3 to start at
   2pm"). Use this whenever the user names an order ID and a new time.

Always use 24-hour "HH:MM" time format for any time argument. Call only
ONE tool per response, even if the user describes multiple changes -
handle them one at a time across multiple turns.

When the user describes a change (e.g. "Machine 2 is down from 10am to 1pm",
"Worker B is out sick on Wednesday afternoon", Machine 3 is not availabe the entire day),
call the `update_availability_and_reschedule` tool with the correct structured
arguments. Always use 24-hour "HH:MM" time format for start_time/end_time.
If time format is not mentioned, identify the timings based on the description
like 'full day' meaning start_time to end_time of the entity,
'morning' means start_time to 12:00; 'after noon' means 12:00 to 16:00 and
'evening' means 16:00 to end_time.
Take start_time and end_time from the SHEDULER_OUTPUT.
identifier can be either an ID (like "M2") or a name (like "Machine 2") -
the tool will resolve it.

Both tools will pause and ask the user to confirm the exact parsed values
before anything is changed. After a tool finishes, summarize in plain
language what changed (or that the user declined) and the resulting
schedule. Only report facts returned by the tool - never invent numbers.
If the user's request doesn't match either tool (e.g. it's not about
availability or a specific order's timing), say so plainly instead of
guessing.
"""


# ---------------------------------------------------------------------
# Availability-editing helpers. These only ever read/write plain dicts -
# they never touch CP-SAT or scheduling logic.
# ---------------------------------------------------------------------
def _set_windows(entity: dict, single_key: str, list_key: str, windows):
    entity.pop(single_key, None)
    entity[list_key] = windows


def _subtract_interval(windows_hours, remove_start, remove_end):
    """Subtract [remove_start, remove_end) from a list of (start, end) hour windows."""
    result = []
    for w_start, w_end in windows_hours:
        if remove_end <= w_start or remove_start >= w_end:
            result.append((w_start, w_end))  # no overlap
            continue
        if remove_start > w_start:
            result.append((w_start, remove_start))
        if remove_end < w_end:
            result.append((remove_end, w_end))
    return result


def _find_entity(data: dict, entity_type: str, identifier: str) -> Optional[dict]:
    collection = data["machines"] if entity_type == "machine" else data["workers"]
    identifier_lower = identifier.strip().lower()
    for e in collection:
        if e["id"].lower() == identifier_lower or e["name"].lower() == identifier_lower:
            return e
    for e in collection:  # loose fallback, e.g. "machine 2" -> "Machine 2"
        if identifier_lower in e["name"].lower() or identifier_lower in e["id"].lower():
            return e
    return None


def _is_affirmative(response: str) -> bool:
    return response.strip().lower() in ("yes", "y", "confirm", "ok", "okay", "go ahead", "proceed", "yep", "sure")


# ---------------------------------------------------------------------
# Tool input schema - lets the LLM call the tool with structured
# arguments instead of us parsing free text ourselves.
# ---------------------------------------------------------------------
class AvailabilityChangeInput(BaseModel):
    entity_type: Literal["machine", "worker"] = Field(
        ..., description="Whether the change applies to a machine or a worker."
    )
    identifier: str = Field(
        ..., description="Machine or worker ID or name, e.g. 'M2' or 'Machine 2'."
    )
    start_time: str = Field(..., description="Start of the unavailable window, 24-hour 'HH:MM'.")
    end_time: str = Field(..., description="End of the unavailable window, 24-hour 'HH:MM'.")


class RescheduleAgent:
    """
    Holds the current schedule state (input data + latest OR-Tools output)
    and exposes a ReAct agent that can modify availability and reschedule
    in response to natural-language instructions - pausing for human
    confirmation of the parsed values before applying anything.
    """

    def __init__(self, initial_data: dict, initial_output: Optional[dict] = None,
            thread_id: str = "reschedule-session"):
        self.current_data = copy.deepcopy(initial_data)
        self.current_output = initial_output
        self.llm = create_llm(temperature=0)
        self._tool = self._build_tool()

        # A checkpointer is REQUIRED for interrupt()/Command(resume=...) to
        # work - it's what lets the graph pause mid-run and be resumed
        # later with the same in-progress state. MemorySaver keeps this in
        # process memory, which is fine for one Streamlit session; swap for
        # a persistent checkpointer later if you need this to survive
        # server restarts.
        self._checkpointer = MemorySaver()
        agent_llm = self.llm.bind(parallel_tool_calls=False)   # <-- only here, tools exist here
        self._graph = create_react_agent(
            agent_llm, [self._tool], prompt=AGENT_SYSTEM_PROMPT, checkpointer=self._checkpointer
        )
        self._config = {"configurable": {"thread_id": thread_id}}

    def _build_tool(self) -> StructuredTool:
        def _run(entity_type: str, identifier: str, start_time: str, end_time: str) -> str:
            data = copy.deepcopy(self.current_data)
            entity = _find_entity(data, entity_type, identifier)
            if entity is None:
                return f"Could not find a {entity_type} matching '{identifier}'. No changes made."

            single_key, list_key = (
                ("available", "available_windows") if entity_type == "machine"
                else ("shift", "shift_windows")
            )
            current_windows = get_windows(entity, single_key, list_key)
            current_hours = [(to_hour_offset(s), to_hour_offset(e)) for s, e in current_windows]

            remove_start = to_hour_offset(start_time)
            remove_end = to_hour_offset(end_time)
            new_hours = _subtract_interval(current_hours, remove_start, remove_end)

            if not new_hours:
                return (f"Marking {entity['name']} unavailable {start_time}-{end_time} would leave "
                        f"zero available time. No changes made - please review the request.")

            new_windows = [[hour_offset_to_clock(s), hour_offset_to_clock(e)] for s, e in new_hours]
            windows_desc = ", ".join(f"{s}-{e}" for s, e in new_windows)

            # -----------------------------------------------------------
            # HUMAN-IN-THE-LOOP: pause here and surface the exact parsed
            # values before writing anything or calling OR-Tools. The
            # graph run stops at this point; RescheduleAgent.request_change
            # returns a "confirmation_required" status with this payload,
            # and nothing below this line runs until confirm_change() is
            # called with the human's reply.
            # -----------------------------------------------------------
            user_reply = interrupt({
                "action": "confirm_availability_change",
                "entity_type": entity_type,
                "entity_id": entity["id"],
                "entity_name": entity["name"],
                "unavailable_start": start_time,
                "unavailable_end": end_time,
                "new_availability_windows": new_windows,
                "prompt": (
                    f"Confirm: mark {entity['name']} unavailable {start_time}-{end_time}? "
                    f"New availability would be: {windows_desc}. Reply yes/no."
                ),
            })

            if not _is_affirmative(user_reply):
                return f"User declined the change ('{user_reply}'). No changes made, schedule unchanged."

            _set_windows(entity, single_key, list_key, new_windows)

            # Persist the updated input JSON, as requested - this is the
            # file OR-Tools is (re)run against.
            with open(UPDATED_DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Re-run the EXISTING, unmodified OR-Tools scheduling logic.
            new_data, new_output = create_schedule(data)
            self.current_data = new_data
            self.current_output = new_output

            stats = new_output["statistics"]
            return (
                f"Confirmed. {entity['name']} marked unavailable {start_time}-{end_time}. "
                f"New availability: {windows_desc}. Rescheduled with OR-Tools: "
                f"{stats['scheduled_count']} of {stats['total_orders']} orders scheduled, "
                f"{stats['unscheduled_count']} unscheduled, {stats['late_count']} late."
            )

        return StructuredTool.from_function(
            func=_run,
            name="update_availability_and_reschedule",
            description=(
                "Mark a machine or worker unavailable for a time range, ask the user "
                "to confirm the exact values, then (if confirmed) write the updated "
                "schedule input to data/factory_data_updated.json and re-run the "
                "OR-Tools scheduler on it. Use this for any request about a machine "
                "going down, being under maintenance, or a worker being unavailable "
                "for part of a day."
            ),
            args_schema=AvailabilityChangeInput,
        )

    def _extract_result(self, result: dict) -> dict:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return {"status": "confirmation_required", "payload": interrupts[0].value}
        final_message = result["messages"][-1].content
        return {
            "status": "done",
            "message": final_message,
            "data": self.current_data,
            "output": self.current_output,
        }

    def request_change(self, instruction: str) -> dict:
        """
        Start a new change request. Returns either:
          {"status": "confirmation_required", "payload": {...}}  -> call
              confirm_change(...) next with the user's yes/no reply
          {"status": "done", "message": ..., "data": ..., "output": ...}
        """
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": instruction}]}, config=self._config
        )
        return self._extract_result(result)

    def confirm_change(self, user_reply: str) -> dict:
        """Resume a paused change request with the human's yes/no (or other) reply."""
        result = self._graph.invoke(Command(resume=user_reply), config=self._config)
        return self._extract_result(result)
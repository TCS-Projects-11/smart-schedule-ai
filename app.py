import json

import streamlit as st

from scheduler import create_schedule
from llm.assistant import SchedulingAssistant

st.set_page_config(page_title="Factory Scheduler", layout="wide")
st.title("Factory Production Scheduler")

# ---------------------------------------------------------------------
# Dynamic input: let the user upload their own factory_data.json instead
# of always solving the bundled sample file.
# ---------------------------------------------------------------------
st.sidebar.header("Schedule Input")
uploaded_file = st.sidebar.file_uploader("Upload factory data (JSON)", type="json")

REQUIRED_KEYS = {"workers", "machines", "orders", "constraints"}


def _validate(data: dict) -> str | None:
    """Return an error message if the JSON is missing required structure, else None."""
    if not isinstance(data, dict):
        return "File must contain a single JSON object."
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return f"Missing required key(s): {', '.join(sorted(missing))}"
    for key in ("workers", "machines", "orders"):
        if not isinstance(data[key], list) or not data[key]:
            return f"'{key}' must be a non-empty list."
    return None


if uploaded_file is not None:
    try:
        new_data = json.load(uploaded_file)
    except json.JSONDecodeError as exc:
        st.sidebar.error(f"Invalid JSON: {exc}")
        st.stop()

    error = _validate(new_data)
    if error:
        st.sidebar.error(f"Invalid schedule data: {error}")
        st.stop()

    # Only treat this as a NEW input if it's actually different from what's
    # currently loaded (by content, not just presence of an uploaded_file),
    # so re-rendering the page doesn't keep re-solving on every interaction.
    if st.session_state.get("uploaded_data") != new_data:
        st.session_state.uploaded_data = new_data
        st.session_state.pop("schedule_result", None)   # force re-solve below
        st.session_state.pop("assistant", None)
        st.session_state.pop("summary", None)
        st.session_state.pop("chat_history", None)
        st.session_state.pop("reschedule_agent", None)
        st.session_state.pop("pending_confirmation", None)

    st.sidebar.success(
        f"Using uploaded data: {len(new_data['workers'])} workers, "
        f"{len(new_data['machines'])} machines, {len(new_data['orders'])} orders."
    )
    active_data = st.session_state.uploaded_data
else:
    active_data = None  # falls back to scheduler.py's bundled DATA_PATH
    if "uploaded_data" in st.session_state:
        st.sidebar.info("Using previously uploaded data. Upload a new file to replace it.")
        active_data = st.session_state.uploaded_data

# ---------------------------------------------------------------------
# Run the solver once per input, cache the result so re-running the UI
# (e.g. asking a question) doesn't re-solve the CP-SAT model.
# ---------------------------------------------------------------------
if "schedule_result" not in st.session_state or st.sidebar.button("Re-run schedule"):
    with st.spinner("Solving schedule..."):
        scheduler_input, scheduler_output = create_schedule(active_data)
    st.session_state.schedule_result = (scheduler_input, scheduler_output)
    st.session_state.pop("assistant", None)
    st.session_state.pop("summary", None)
    st.session_state.pop("chat_history", None)
    st.session_state.pop("reschedule_agent", None)
    st.session_state.pop("pending_confirmation", None)

scheduler_input, scheduler_output = st.session_state.schedule_result
stats = scheduler_output["statistics"]

# ---------------------------------------------------------------------
# Top-line stats
# ---------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total orders", stats["total_orders"])
col2.metric("Scheduled", stats["scheduled_count"])
col3.metric("Unscheduled", stats["unscheduled_count"])
col4.metric("Late", stats["late_count"])

# ---------------------------------------------------------------------
# Scheduled orders table
# ---------------------------------------------------------------------
st.subheader("Scheduled Orders")
if scheduler_output["scheduled_orders"]:
    st.dataframe(scheduler_output["scheduled_orders"], use_container_width=True)
else:
    st.info("No orders were scheduled.")

# ---------------------------------------------------------------------
# Unscheduled orders table
# ---------------------------------------------------------------------
st.subheader("Unscheduled Orders")
if scheduler_output["unscheduled_orders"]:
    st.dataframe(scheduler_output["unscheduled_orders"], use_container_width=True)
else:
    st.success("Every order was scheduled.")

# ---------------------------------------------------------------------
# Machine utilization / worker workload
# ---------------------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Machine Utilization")
    st.dataframe(scheduler_output["machine_utilization"], use_container_width=True)
with col_b:
    st.subheader("Worker Workload")
    st.dataframe(scheduler_output["worker_workload"], use_container_width=True)

# ---------------------------------------------------------------------
# LLM: automatic summary
# ---------------------------------------------------------------------
st.divider()
st.header("AI Analysis")

if "assistant" not in st.session_state:
    try:
        st.session_state.assistant = SchedulingAssistant(
            scheduler_input=scheduler_input,
            scheduler_output=scheduler_output,
        )
    except Exception as exc:
        st.session_state.assistant = None
        st.error(f"LLM assistant could not start: {exc}")

assistant = st.session_state.get("assistant")

if assistant:
    st.subheader("Summary")
    if "summary" not in st.session_state:
        with st.spinner("Generating summary..."):
            st.session_state.summary = assistant.generate_summary()
    st.write(st.session_state.summary)

    # -------------------------------------------------------------
    # LLM: Reschedule
    # -------------------------------------------------------------
    st.divider()
    st.header("Update Schedule")

    if "reschedule_agent" not in st.session_state:
        from llm.reschedule_agent import RescheduleAgent
        st.session_state.reschedule_agent = RescheduleAgent(scheduler_input, scheduler_output)

    pending = st.session_state.get("pending_confirmation")

    if pending:
        st.warning(pending["prompt"])
        col_yes, col_no = st.columns(2)
        if col_yes.button("Confirm"):
            result = st.session_state.reschedule_agent.confirm_change("yes")
            st.session_state.pending_confirmation = None
            if result["status"] == "done":
                st.session_state.schedule_result = (result["data"], result["output"])
                st.session_state.pop("assistant", None)
                st.session_state.pop("summary", None)
                st.success(result["message"])
            st.rerun()
        if col_no.button("Cancel"):
            result = st.session_state.reschedule_agent.confirm_change("no")
            st.session_state.pending_confirmation = None
            st.info(result["message"])
            st.rerun()
    else:
        change_request = st.text_input(
            "Describe a change",
            placeholder="e.g. Machine 2 will not be available between 10am and 1pm.",
        )
        if st.button("Apply change") and change_request:
            with st.spinner("Parsing request..."):
                result = st.session_state.reschedule_agent.request_change(change_request)
            if result["status"] == "confirmation_required":
                st.session_state.pending_confirmation = result["payload"]
            else:
                st.session_state.schedule_result = (result["data"], result["output"])
                st.session_state.pop("assistant", None)
                st.session_state.pop("summary", None)
                st.success(result["message"])
            st.rerun()

    # -------------------------------------------------------------
    # LLM: Q&A
    # -------------------------------------------------------------
    st.subheader("Ask about the schedule")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    question = st.chat_input("e.g. Why was O5 not scheduled?")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            answer = assistant.answer_question(question)
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((question, answer))

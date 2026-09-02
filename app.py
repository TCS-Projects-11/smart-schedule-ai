"""
app.py
-------
Minimal Streamlit UI over the existing scheduler + LLM layer.
Does not modify scheduler.py or llm/* - it only calls their public
functions/classes (create_schedule, SchedulingAssistant).
"""

import streamlit as st

from scheduler import create_schedule
from llm.assistant import SchedulingAssistant

st.set_page_config(page_title="Factory Scheduler", layout="wide")
st.title("Factory Production Scheduler")

# ---------------------------------------------------------------------
# Run the solver once per session, cache the result so re-running the
# UI (e.g. asking a question) doesn't re-solve the CP-SAT model.
# Click "Re-run schedule" to force a fresh solve.
# ---------------------------------------------------------------------
if "schedule_result" not in st.session_state or st.sidebar.button("Re-run schedule"):
    with st.spinner("Solving schedule..."):
        scheduler_input, scheduler_output = create_schedule()
    st.session_state.schedule_result = (scheduler_input, scheduler_output)
    st.session_state.pop("assistant", None)      # force LLM re-init on new data
    st.session_state.pop("summary", None)
    st.session_state.pop("chat_history", None)

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
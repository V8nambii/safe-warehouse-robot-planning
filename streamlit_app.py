from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from warehouse_llm.llm import OllamaClient
from warehouse_llm.parsing import parse_direction, parse_mission_plan, parse_route
from warehouse_llm.planner import validate_mission, validate_path
from warehouse_llm.prompts import build_direction_prompt, build_mission_prompt, build_route_prompt
from warehouse_llm.simulator import DIFFICULTY_CONFIG, WarehouseSimulator
from warehouse_llm.visualization import (
    plot_direction_scenario,
    plot_mission_scenario,
    plot_route_scenario,
)


st.set_page_config(page_title="Warehouse LLM Lab", page_icon="🤖", layout="wide")
st.title("LLM-Assisted Warehouse Task and Route Planning Lab")
st.caption("Generate pick-and-deliver jobs, test an LLM, and verify every route against symbolic ground truth.")

simulator = WarehouseSimulator()
with st.sidebar:
    st.header("Scenario settings")
    task = st.radio("Task", ("Pick-and-deliver mission", "Route planning", "Direction reasoning"))
    difficulty = st.select_slider("Difficulty", options=list(DIFFICULTY_CONFIG), value="medium")
    seed = st.number_input("Random seed", min_value=0, value=2026, step=1)
    style_options = (
        ("minimal", "rules", "few_shot", "distractor_map")
        if task == "Direction reasoning"
        else ("minimal", "rules", "few_shot")
    )
    prompt_style = st.selectbox("Prompt style", style_options, index=min(1, len(style_options) - 1))
    st.divider()
    st.header("Ollama connection")
    model = st.text_input("Model", "qwen:7b")
    base_url = st.text_input("Server", "http://127.0.0.1:11434")

tab_demo, tab_results, tab_about = st.tabs(("Live demonstration", "Analyse result CSV", "Research design"))

with tab_demo:
    left, right = st.columns((1.05, 0.95))
    if task == "Direction reasoning":
        scenario = simulator.generate_direction_scenario(int(seed), difficulty)
        prompt = build_direction_prompt(scenario, simulator.warehouse, prompt_style)
        with left:
            st.pyplot(plot_direction_scenario(simulator.warehouse, scenario), clear_figure=True)
        with right:
            st.subheader("Generated task")
            st.code(prompt, language="text")
            st.metric("Symbolic ground truth", scenario.expected.value)
            if st.button("Ask Qwen", type="primary"):
                try:
                    with st.spinner("Waiting for the model..."):
                        response = OllamaClient(model=model, base_url=base_url).generate(prompt, scenario)
                    parsed = parse_direction(response.text)
                    st.write("Raw response:", response.text)
                    st.success("Correct") if parsed.parsed == scenario.expected else st.error("Incorrect")
                    st.write(
                        {
                            "parsed": parsed.parsed.value if parsed.parsed else None,
                            "format_compliant": parsed.contract_valid,
                            "latency_ms": round(response.latency_ms, 1),
                        }
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
    elif task == "Route planning":
        scenario = simulator.generate_route_scenario(int(seed), difficulty)
        prompt = build_route_prompt(scenario, simulator.warehouse, prompt_style)
        with left:
            st.pyplot(plot_route_scenario(simulator.warehouse, scenario), clear_figure=True)
        with right:
            st.subheader("Generated task")
            st.code(prompt, language="text")
            st.metric("A* optimal route length", f"{scenario.optimal_steps} steps")
            with st.expander("Show optimal coordinates"):
                st.code(json.dumps([point.as_list() for point in scenario.optimal_path]))
            if st.button("Ask Qwen", type="primary"):
                try:
                    with st.spinner("Waiting for the model..."):
                        response = OllamaClient(model=model, base_url=base_url).generate(prompt, scenario)
                    path = parse_route(response.text)
                    check = validate_path(simulator.warehouse, path, scenario.start, scenario.goal)
                    st.write("Raw response:", response.text)
                    st.success("Valid route") if check.valid else st.error("Invalid route")
                    st.write(check.__dict__)
                    if path:
                        st.pyplot(
                            plot_route_scenario(simulator.warehouse, scenario, path),
                            clear_figure=True,
                        )
                except RuntimeError as exc:
                    st.error(str(exc))
    else:
        scenario = simulator.generate_mission_scenario(int(seed), difficulty)
        prompt = build_mission_prompt(scenario, simulator.warehouse, prompt_style)
        with left:
            st.pyplot(plot_mission_scenario(simulator.warehouse, scenario), clear_figure=True)
        with right:
            st.subheader("Worker instruction")
            st.info(scenario.instruction)
            st.code(prompt, language="text")
            metric_a, metric_b = st.columns(2)
            metric_a.metric("Requested item", scenario.item_id)
            metric_b.metric("Destination", scenario.destination_id)
            st.metric("A* optimal mission", f"{scenario.optimal_steps} steps")
            with st.expander("Show optimal mission coordinates"):
                st.code(json.dumps([point.as_list() for point in scenario.optimal_path]))
            if st.button("Ask Qwen", type="primary"):
                try:
                    with st.spinner("Waiting for the model..."):
                        response = OllamaClient(model=model, base_url=base_url).generate(prompt, scenario)
                    parsed = parse_mission_plan(response.text)
                    plan = parsed.plan
                    check = validate_mission(
                        simulator.warehouse, plan.route if plan else None, scenario
                    )
                    task_correct = bool(
                        plan
                        and plan.action == "pick_and_deliver"
                        and plan.item_id.casefold() == scenario.item_id.casefold()
                        and plan.destination_id.casefold() == scenario.destination_id.casefold()
                    )
                    valid = task_correct and check.valid_route_for_mission
                    st.write("Raw response:", response.text)
                    st.success("Valid pick-and-deliver mission") if valid else st.error("Invalid mission")
                    st.write(
                        {
                            "task_correct": task_correct,
                            "route_valid": check.route.valid,
                            "pickup_visited": check.pickup_visited,
                            "format_compliant": parsed.contract_valid,
                            "latency_ms": round(response.latency_ms, 1),
                        }
                    )
                    if plan:
                        st.pyplot(
                            plot_mission_scenario(simulator.warehouse, scenario, plan.route),
                            clear_figure=True,
                        )
                except RuntimeError as exc:
                    st.error(str(exc))

with tab_results:
    uploaded = st.file_uploader("Upload a CSV produced by the experiment runner", type="csv")
    if uploaded:
        frame = pd.read_csv(uploaded)
        st.dataframe(frame, use_container_width=True)
        if "correct" in frame.columns:
            success_column = "correct"
        elif "valid_mission" in frame.columns:
            success_column = "valid_mission"
        else:
            success_column = "valid_route"
        if success_column in frame.columns:
            frame[success_column] = frame[success_column].astype(str).str.lower().eq("true")
            chart = (
                frame.groupby(["difficulty", "prompt_style"], as_index=False)[success_column]
                .mean()
                .rename(columns={success_column: "success_rate"})
            )
            st.subheader("Success rate by condition")
            st.bar_chart(chart, x="difficulty", y="success_rate", color="prompt_style", stack=False)

with tab_about:
    st.markdown(
        """
        **Primary research question:** How reliably can a small language model convert warehouse
        instructions into correct pick-and-deliver tasks and safe routes?

        **Independent variables:** task difficulty, prompt style and model.

        **Dependent variables:** task-understanding accuracy, pickup completion, direction accuracy,
        output-format compliance, valid-route rate, collision rate, path efficiency and response time.

        The Python simulator and A* planner provide deterministic ground truth. The LLM is never
        treated as ground truth and is not allowed to control a physical robot.
        """
    )

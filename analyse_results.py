from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]
EXPECTED_DESIGN = {
    "direction": {
        "prompt_styles": ["minimal", "rules", "few_shot", "distractor_map"],
        "cases_per_condition": 30,
    },
    "route": {
        "prompt_styles": ["minimal", "rules", "few_shot"],
        "cases_per_condition": 20,
    },
    "mission": {
        "prompt_styles": ["minimal", "rules", "few_shot"],
        "cases_per_condition": 20,
    },
    "hybrid": {
        "prompt_styles": ["minimal", "rules", "few_shot"],
        "cases_per_condition": 20,
    },
}
METRICS = {
    "direction": [
        ("correct", "Direction accuracy", "primary"),
        ("strict_correct", "Strict direction accuracy", "diagnostic"),
        ("format_compliant", "One-label format compliance", "diagnostic"),
    ],
    "route": [
        ("valid_route", "Valid-route rate", "primary"),
        ("optimal_route", "Optimal-route rate", "diagnostic"),
        ("parseable", "Parseable route rate", "diagnostic"),
        ("starts_correctly", "Correct-start rate", "diagnostic"),
        ("reaches_goal", "Goal-reaching rate", "diagnostic"),
        ("moves_are_adjacent", "Adjacent-move rate", "diagnostic"),
        ("collision_free", "Collision-free rate", "diagnostic"),
    ],
    "mission": [
        ("valid_mission", "Valid-mission rate", "primary"),
        ("task_correct", "Semantic task accuracy", "diagnostic"),
        ("optimal_mission", "Optimal-mission rate", "diagnostic"),
        ("format_compliant", "JSON/schema compliance", "diagnostic"),
        ("pickup_visited", "Pickup-visit rate", "diagnostic"),
        ("parseable_route", "Parseable route rate", "diagnostic"),
        ("starts_correctly", "Correct-start rate", "diagnostic"),
        ("reaches_destination", "Destination-reaching rate", "diagnostic"),
        ("moves_are_adjacent", "Adjacent-move rate", "diagnostic"),
        ("collision_free", "Collision-free rate", "diagnostic"),
    ],
    "hybrid": [
        ("valid_hybrid_mission", "Valid hybrid-mission rate", "primary"),
        ("task_correct", "Semantic task accuracy", "diagnostic"),
        ("parseable_task", "Parseable task rate", "diagnostic"),
        ("format_compliant", "JSON/schema compliance", "diagnostic"),
        ("route_generated", "A* route-generation rate", "diagnostic"),
        ("symbolic_route_valid", "Generated-route validity", "diagnostic"),
        ("pickup_visited", "Correct pickup-visit rate", "diagnostic"),
        ("reaches_destination", "Correct destination-reaching rate", "diagnostic"),
        ("optimal_hybrid_mission", "Optimal hybrid-mission rate", "diagnostic"),
    ],
}
PRIMARY_METRIC = {
    "direction": "correct",
    "route": "valid_route",
    "mission": "valid_mission",
    "hybrid": "valid_hybrid_mission",
}
TASK_LABEL = {
    "direction": "Direction reasoning",
    "route": "Route planning",
    "mission": "Pick-and-deliver mission",
    "hybrid": "Hybrid LLM + A* mission",
}


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert CSV booleans safely; missing values count as False."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.casefold().eq("true")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binary proportion."""
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + (z**2 / total)
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def aggregate(frame: pd.DataFrame, success_column: str) -> pd.DataFrame:
    """Summarise a binary outcome for every model/difficulty/prompt condition."""
    rows = []
    for keys, group in frame.groupby(["model", "difficulty", "prompt_style"], dropna=False):
        successes = int(_as_bool(group[success_column]).sum())
        total = len(group)
        low, high = wilson_interval(successes, total)
        rows.append(
            {
                "model": keys[0],
                "difficulty": keys[1],
                "prompt_style": keys[2],
                "successes": successes,
                "n": total,
                "rate": successes / total,
                "rate_percent": round(100 * successes / total, 2),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def _inference_error_mask(frame: pd.DataFrame) -> pd.Series:
    explicit = pd.Series(False, index=frame.index)
    if "inference_error" in frame:
        explicit = frame["inference_error"].fillna("").astype(str).str.strip().ne("")
    raw = pd.Series("", index=frame.index)
    if "raw_response" in frame:
        raw = frame["raw_response"].fillna("").astype(str)
    return explicit | raw.str.startswith("INFERENCE_ERROR")


def integrity_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Check that the final files match the preregistered sample structure."""
    rows = []
    for task, design in EXPECTED_DESIGN.items():
        frame = combined[combined["task"] == task].copy()
        if task == "hybrid" and frame.empty:
            continue
        expected_conditions = {
            (difficulty, style)
            for difficulty in DIFFICULTY_ORDER
            for style in design["prompt_styles"]
        }
        observed_conditions = set(
            zip(frame.get("difficulty", []), frame.get("prompt_style", []), strict=False)
        )
        expected_rows = len(expected_conditions) * design["cases_per_condition"]
        unique_scenarios = (
            int(frame["scenario_id"].nunique(dropna=False))
            if "scenario_id" in frame
            else 0
        )
        duplicate_rows = max(0, len(frame) - unique_scenarios)
        condition_sizes = frame.groupby(["difficulty", "prompt_style"]).size()
        complete = (
            len(frame) == expected_rows
            and duplicate_rows == 0
            and observed_conditions == expected_conditions
            and not condition_sizes.empty
            and condition_sizes.eq(design["cases_per_condition"]).all()
        )
        error_count = int(_inference_error_mask(frame).sum()) if not frame.empty else 0
        if complete and error_count == 0:
            status = "PASS"
        elif complete:
            status = "PASS_WITH_RECORDED_INFERENCE_ERRORS"
        else:
            status = "CHECK_DESIGN"
        rows.append(
            {
                "task": task,
                "expected_cases": expected_rows,
                "observed_cases": len(frame),
                "unique_scenarios": unique_scenarios,
                "duplicate_scenario_rows": duplicate_rows,
                "expected_conditions": len(expected_conditions),
                "observed_conditions": len(observed_conditions),
                "cases_per_condition_min": int(condition_sizes.min()) if len(condition_sizes) else 0,
                "cases_per_condition_max": int(condition_sizes.max()) if len(condition_sizes) else 0,
                "inference_errors": error_count,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def overall_accuracy_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall primary and diagnostic rates with Wilson intervals."""
    rows = []
    for task, metric_definitions in METRICS.items():
        frame = combined[combined["task"] == task].copy()
        for metric, label, role in metric_definitions:
            if frame.empty or metric not in frame:
                continue
            values = _as_bool(frame[metric])
            successes = int(values.sum())
            total = len(values)
            low, high = wilson_interval(successes, total)
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABEL[task],
                    "metric": metric,
                    "metric_label": label,
                    "metric_role": role,
                    "successes": successes,
                    "n": total,
                    "rate": successes / total,
                    "rate_percent": round(100 * successes / total, 2),
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci95_low_percent": round(100 * low, 2),
                    "ci95_high_percent": round(100 * high, 2),
                    "inference_errors": int(_inference_error_mask(frame).sum()),
                }
            )
    return pd.DataFrame(rows)


def primary_group_summary(combined: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for task, success_column in PRIMARY_METRIC.items():
        frame = combined[combined["task"] == task].copy()
        if frame.empty or success_column not in frame:
            continue
        for group_name, group in frame.groupby(group_column, dropna=False):
            successes = int(_as_bool(group[success_column]).sum())
            total = len(group)
            low, high = wilson_interval(successes, total)
            rows.append(
                {
                    "task": task,
                    group_column: group_name,
                    "metric": success_column,
                    "successes": successes,
                    "n": total,
                    "rate": successes / total,
                    "rate_percent": round(100 * successes / total, 2),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_mission_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    """Compare direct and hybrid systems on the same generated mission cases."""
    direct = combined[combined["task"] == "mission"].copy()
    hybrid = combined[combined["task"] == "hybrid"].copy()
    if direct.empty or hybrid.empty:
        return pd.DataFrame()
    keys = ["scenario_id", "difficulty", "prompt_style", "model"]
    direct = direct[keys + ["valid_mission"]].rename(
        columns={"valid_mission": "direct_success"}
    )
    hybrid = hybrid[keys + ["valid_hybrid_mission"]].rename(
        columns={"valid_hybrid_mission": "hybrid_success"}
    )
    paired = direct.merge(hybrid, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return pd.DataFrame()
    direct_success = _as_bool(paired["direct_success"])
    hybrid_success = _as_bool(paired["hybrid_success"])
    total = len(paired)
    return pd.DataFrame(
        [
            {
                "paired_cases": total,
                "direct_llm_successes": int(direct_success.sum()),
                "direct_llm_rate": float(direct_success.mean()),
                "hybrid_successes": int(hybrid_success.sum()),
                "hybrid_rate": float(hybrid_success.mean()),
                "absolute_improvement_percentage_points": round(
                    100 * float(hybrid_success.mean() - direct_success.mean()), 2
                ),
                "both_succeeded": int((direct_success & hybrid_success).sum()),
                "direct_only_succeeded": int((direct_success & ~hybrid_success).sum()),
                "hybrid_only_succeeded": int((~direct_success & hybrid_success).sum()),
                "both_failed": int((~direct_success & ~hybrid_success).sum()),
            }
        ]
    )


def plot_success(summary: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x_lookup = {name: index for index, name in enumerate(DIFFICULTY_ORDER)}
    for (model, style), group in summary.groupby(["model", "prompt_style"]):
        ordered = group.copy()
        ordered["x"] = ordered["difficulty"].map(x_lookup)
        ordered = ordered.sort_values("x")
        yerr = [
            ordered["rate"] - ordered["ci95_low"],
            ordered["ci95_high"] - ordered["rate"],
        ]
        ax.errorbar(
            ordered["x"],
            ordered["rate"],
            yerr=yerr,
            marker="o",
            capsize=3,
            label=f"{model} — {style}",
        )
    ax.set_xticks(range(len(DIFFICULTY_ORDER)), [name.title() for name in DIFFICULTY_ORDER])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Success proportion (95% Wilson CI)")
    ax.set_xlabel("Difficulty")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_overview(overall: pd.DataFrame, output: Path) -> None:
    primary = overall[overall["metric_role"] == "primary"].copy()
    if primary.empty:
        return
    labels = primary["task_label"].tolist()
    rates = primary["rate"].tolist()
    positions = list(range(len(rates)))
    errors = [
        (primary["rate"] - primary["ci95_low"]).tolist(),
        (primary["ci95_high"] - primary["rate"]).tolist(),
    ]
    y_max = min(1.0, max(0.12, float(primary["ci95_high"].max()) * 1.6))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    palette = ["#2563EB", "#0F766E", "#C2410C", "#7C3AED"]
    bars = ax.bar(positions, rates, color=palette[: len(rates)], width=0.62)
    ax.errorbar(positions, rates, yerr=errors, fmt="none", ecolor="#111827", capsize=5)
    for bar, row in zip(bars, primary.itertuples(index=False), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(y_max * 0.92, bar.get_height() + y_max * 0.07),
            f"{row.successes}/{row.n}\n{row.rate_percent:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, y_max)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Primary success rate (95% Wilson CI)")
    ax.set_title("Primary performance by warehouse reasoning task")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report_table(overall: pd.DataFrame, role: str) -> pd.DataFrame:
    selected = overall[overall["metric_role"] == role].copy()
    selected["Result"] = selected.apply(
        lambda row: f"{int(row['successes'])}/{int(row['n'])} ({row['rate_percent']:.2f}%)",
        axis=1,
    )
    selected["95% Wilson CI"] = selected.apply(
        lambda row: f"{row['ci95_low_percent']:.2f}%–{row['ci95_high_percent']:.2f}%",
        axis=1,
    )
    return selected[["task_label", "metric_label", "Result", "95% Wilson CI"]].rename(
        columns={"task_label": "Task", "metric_label": "Metric"}
    )


def build_brief_report(
    combined: pd.DataFrame,
    overall: pd.DataFrame,
    integrity: pd.DataFrame,
    condition_summary: pd.DataFrame,
    mission_comparison: pd.DataFrame,
) -> str:
    primary = overall[overall["metric_role"] == "primary"].set_index("task")
    report = [
        "# Brief accuracy report",
        "",
        "## Method",
        "",
        (
            "This report was generated directly from the final CSV files. Each binary outcome is "
            "reported as successes/total, a percentage, and a two-sided 95% Wilson confidence "
            "interval. Failed, malformed and timed-out model responses remain in the denominator."
        ),
        "",
        "## Data-integrity checks",
        "",
        markdown_table(integrity),
        "",
        "## Primary results",
        "",
        markdown_table(_report_table(overall, "primary")),
        "",
    ]
    if not primary.empty:
        for task in ["direction", "route", "mission", "hybrid"]:
            if task not in primary.index:
                continue
            row = primary.loc[task]
            report.append(
                f"- **{TASK_LABEL[task]}:** {int(row['successes'])} of {int(row['n'])} cases "
                f"succeeded ({row['rate_percent']:.2f}%, 95% CI "
                f"{row['ci95_low_percent']:.2f}%–{row['ci95_high_percent']:.2f}%)."
            )
        core_primary = primary.loc[
            primary.index.intersection(["direction", "route", "mission"])
        ]
        total_successes = int(core_primary["successes"].sum())
        total_cases = int(core_primary["n"].sum())
        report.extend(
            [
                "",
                (
                    f"Across the three original heterogeneous tasks, the case-weighted primary "
                    f"success count was "
                    f"{total_successes}/{total_cases} ({100 * total_successes / total_cases:.2f}%). "
                    "This is a descriptive combined rate, not a single interchangeable definition "
                    "of accuracy, because the original tasks use different success criteria and the "
                    "direction task has twice as many cases."
                ),
                "",
            ]
        )
    if not mission_comparison.empty:
        comparison = mission_comparison.iloc[0]
        report.extend(
            [
                "## Direct LLM versus hybrid LLM + A*",
                "",
                (
                    f"The paired comparison matched {int(comparison['paired_cases'])} mission "
                    f"scenarios. Direct LLM planning succeeded on "
                    f"{int(comparison['direct_llm_successes'])} cases "
                    f"({100 * comparison['direct_llm_rate']:.2f}%), whereas the hybrid system "
                    f"succeeded on {int(comparison['hybrid_successes'])} cases "
                    f"({100 * comparison['hybrid_rate']:.2f}%). The descriptive absolute change "
                    f"was {comparison['absolute_improvement_percentage_points']:.2f} percentage "
                    "points."
                ),
                "",
                (
                    "The hybrid is an exploratory extension added after the original results were "
                    "observed. It must be reported separately rather than presented as part of the "
                    "original confirmatory design."
                ),
                "",
            ]
        )
    report.extend(
        [
            "## Diagnostic results",
            "",
            markdown_table(_report_table(overall, "diagnostic")),
            "",
            "## Condition-level observations",
            "",
        ]
    )
    for task in ["direction", "route", "mission", "hybrid"]:
        task_conditions = condition_summary[condition_summary["task"] == task]
        if task_conditions.empty:
            continue
        best_rate = task_conditions["rate"].max()
        best_rows = task_conditions[task_conditions["rate"] == best_rate]
        if len(best_rows) == 1:
            best = best_rows.iloc[0]
            report.append(
                f"- **{TASK_LABEL[task]}:** the highest observed condition was "
                f"{best['difficulty']} difficulty with the {best['prompt_style']} prompt "
                f"({int(best['successes'])}/{int(best['n'])}, {best['rate_percent']:.2f}%)."
            )
        elif len(best_rows) <= 3:
            names = "; ".join(
                f"{row.difficulty}/{row.prompt_style} ({int(row.successes)}/{int(row.n)})"
                for row in best_rows.itertuples(index=False)
            )
            report.append(
                f"- **{TASK_LABEL[task]}:** {len(best_rows)} conditions tied for the highest "
                f"observed rate of {100 * best_rate:.2f}%: {names}."
            )
        else:
            report.append(
                f"- **{TASK_LABEL[task]}:** all {len(best_rows)} conditions tied at the "
                f"highest observed rate of {100 * best_rate:.2f}%."
            )
    report.extend(
        [
            "",
            (
                "These condition comparisons are descriptive. Small condition samples and overlapping "
                "confidence intervals mean that a visual difference alone should not be described as "
                "statistically significant."
            ),
            "",
            "## Interpretation for the dissertation",
            "",
            (
                "The experiment measures spatial reasoning and constrained planning, not physical robot "
                "performance. A valid route must be parseable, start correctly, use adjacent legal moves, "
                "avoid obstacles and reach the goal. A valid mission must also select the requested item "
                "and destination and visit a legal pickup-access cell. The strict end-to-end definitions "
                "explain why route and mission scores can be much lower than JSON parseability or semantic "
                "task accuracy. Low scores are legitimate findings and must not be removed or manually fixed."
            ),
            "",
            (
                "In the hybrid extension, the LLM performs high-level task extraction only. The A* "
                "component generates the route and the symbolic validator checks it before execution. "
                "This separates language-understanding errors from navigation errors."
            ),
            "",
            "## Reproducibility note",
            "",
            (
                "Retain the raw final CSV files, executed notebook, model name and digest, temperature, "
                "scenario seeds, Python/Ollama versions and Mac hardware details. The oracle check validates "
                "the software pipeline and must not be reported as LLM accuracy."
            ),
            "",
        ]
    )
    return "\n".join(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse Warehouse LLM experiment CSV files")
    parser.add_argument("--input-dir", default="results/final")
    parser.add_argument(
        "--hybrid-dir",
        default=None,
        help="Optional directory containing the exploratory hybrid mission CSV",
    )
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.glob("*.csv"))
    if args.hybrid_dir:
        csv_files.extend(sorted(Path(args.hybrid_dir).glob("*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    frames = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "task" not in combined:
        raise SystemExit("The CSV files do not contain a 'task' column.")

    integrity = integrity_summary(combined)
    integrity.to_csv(output_dir / "experiment_integrity.csv", index=False)
    (output_dir / "experiment_integrity.json").write_text(
        json.dumps(integrity.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    if int(integrity["duplicate_scenario_rows"].sum()) > 0:
        raise SystemExit(
            "Duplicate scenario IDs were found. See experiment_integrity.csv; do not report "
            "accuracy until the repeated runs are resolved and documented."
        )

    overall = overall_accuracy_summary(combined)
    overall.to_csv(output_dir / "overall_accuracy_summary.csv", index=False)
    plot_accuracy_overview(overall, output_dir / "accuracy_overview.png")

    by_difficulty = primary_group_summary(combined, "difficulty")
    by_difficulty.to_csv(output_dir / "primary_accuracy_by_difficulty.csv", index=False)
    by_prompt = primary_group_summary(combined, "prompt_style")
    by_prompt.to_csv(output_dir / "primary_accuracy_by_prompt_style.csv", index=False)

    condition_frames = []
    report_lines = ["# Detailed experimental results", ""]

    direction = combined[combined["task"] == "direction"].copy()
    if not direction.empty:
        summary = aggregate(direction, "correct")
        summary.insert(0, "task", "direction")
        condition_frames.append(summary)
        summary.to_csv(output_dir / "direction_summary_with_ci.csv", index=False)
        plot_success(summary, "LLM direction accuracy", output_dir / "direction_accuracy.png")
        report_lines.extend(["## Direction reasoning", "", markdown_table(summary), ""])
        confusion = pd.crosstab(
            direction["expected"], direction["parsed_response"], dropna=False, margins=True
        )
        confusion.to_csv(output_dir / "direction_confusion_matrix.csv")

    route = combined[combined["task"] == "route"].copy()
    if not route.empty:
        summary = aggregate(route, "valid_route")
        summary.insert(0, "task", "route")
        condition_frames.append(summary)
        summary.to_csv(output_dir / "route_summary_with_ci.csv", index=False)
        plot_success(summary, "LLM valid-route rate", output_dir / "valid_route_rate.png")
        report_lines.extend(["## Route planning", "", markdown_table(summary), ""])
        failure_columns = [
            "parseable",
            "starts_correctly",
            "reaches_goal",
            "moves_are_adjacent",
            "collision_free",
        ]
        failure_rows = [
            {"criterion": column, "failure_count": int((~_as_bool(route[column])).sum())}
            for column in failure_columns
        ]
        pd.DataFrame(failure_rows).to_csv(output_dir / "route_failure_counts.csv", index=False)

    mission = combined[combined["task"] == "mission"].copy()
    if not mission.empty:
        summary = aggregate(mission, "valid_mission")
        summary.insert(0, "task", "mission")
        condition_frames.append(summary)
        summary.to_csv(output_dir / "mission_summary_with_ci.csv", index=False)
        plot_success(
            summary,
            "Valid pick-and-deliver mission rate",
            output_dir / "valid_mission_rate.png",
        )
        report_lines.extend(
            ["## Pick-and-deliver task planning", "", markdown_table(summary), ""]
        )
        failure_columns = [
            "format_compliant",
            "action_correct",
            "item_correct",
            "destination_correct",
            "parseable_route",
            "starts_correctly",
            "reaches_destination",
            "moves_are_adjacent",
            "collision_free",
            "pickup_visited",
        ]
        failure_rows = [
            {"criterion": column, "failure_count": int((~_as_bool(mission[column])).sum())}
            for column in failure_columns
        ]
        pd.DataFrame(failure_rows).to_csv(output_dir / "mission_failure_counts.csv", index=False)

    hybrid = combined[combined["task"] == "hybrid"].copy()
    if not hybrid.empty:
        summary = aggregate(hybrid, "valid_hybrid_mission")
        summary.insert(0, "task", "hybrid")
        condition_frames.append(summary)
        summary.to_csv(output_dir / "hybrid_summary_with_ci.csv", index=False)
        plot_success(
            summary,
            "Hybrid LLM + A* valid-mission rate",
            output_dir / "valid_hybrid_mission_rate.png",
        )
        report_lines.extend(
            ["## Hybrid LLM + A* mission planning", "", markdown_table(summary), ""]
        )
        failure_columns = [
            "parseable_task",
            "format_compliant",
            "action_correct",
            "item_correct",
            "destination_correct",
            "task_correct",
            "route_generated",
            "symbolic_route_valid",
            "pickup_visited",
            "reaches_destination",
        ]
        failure_rows = [
            {"criterion": column, "failure_count": int((~_as_bool(hybrid[column])).sum())}
            for column in failure_columns
        ]
        pd.DataFrame(failure_rows).to_csv(output_dir / "hybrid_failure_counts.csv", index=False)

    condition_summary = pd.concat(condition_frames, ignore_index=True, sort=False)
    condition_summary.to_csv(output_dir / "condition_accuracy_summary.csv", index=False)
    mission_comparison = paired_mission_comparison(combined)
    if not mission_comparison.empty:
        mission_comparison.to_csv(
            output_dir / "direct_vs_hybrid_mission_comparison.csv", index=False
        )
    (output_dir / "results_summary.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    brief_report = build_brief_report(
        combined, overall, integrity, condition_summary, mission_comparison
    )
    (output_dir / "brief_accuracy_report.md").write_text(brief_report, encoding="utf-8")

    primary = overall[overall["metric_role"] == "primary"]
    print("\nPRIMARY ACCURACY RESULTS")
    for row in primary.itertuples(index=False):
        print(
            f"{row.task_label}: {row.successes}/{row.n} = {row.rate_percent:.2f}% "
            f"(95% CI {row.ci95_low_percent:.2f}% to {row.ci95_high_percent:.2f}%)"
        )
    print(f"\nAnalysis saved in {output_dir.resolve()}")
    print("Open brief_accuracy_report.md for the dissertation-ready summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

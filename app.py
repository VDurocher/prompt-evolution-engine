"""
Streamlit interface for the Prompt Evolution Engine.
Launches the evolution and displays results in real time.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

import config as cfg
from core.evaluator import EvalCriteria
from core.evolution import EvolutionConfig, EvolutionEngine, EvolutionState
from core.genome import PromptGenome

# --- Page configuration ---
st.set_page_config(
    page_title="Prompt Evolution Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .technique-badge {
        display: inline-block;
        background: #313244;
        color: #cdd6f4;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .best-prompt {
        background: #1e2a1e;
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Visualisation helpers ──────────────────────────────────────────────


def build_score_chart(state: EvolutionState) -> go.Figure:
    """Plotly chart of score evolution per generation."""
    results = state.generation_results
    gens = [r.generation for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=[r.max_score for r in results],
        mode="lines+markers", name="Max",
        line={"color": "#2ecc71", "width": 2},
        marker={"size": 8, "symbol": "circle"},
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=[r.mean_score for r in results],
        mode="lines+markers", name="Mean",
        line={"color": "#f1c40f", "width": 2, "dash": "dot"},
        marker={"size": 6},
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=[r.min_score for r in results],
        mode="lines+markers", name="Min",
        line={"color": "#e74c3c", "width": 1, "dash": "dash"},
        marker={"size": 5},
    ))

    fig.update_layout(
        title={"text": "Score per generation", "font": {"size": 16}},
        xaxis_title="Generation",
        yaxis_title="Score (0 → 1)",
        yaxis={"range": [0, 1.05], "gridcolor": "#313244"},
        xaxis={"gridcolor": "#313244", "tickmode": "linear"},
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cdd6f4"},
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 40, "b": 20},
    )
    return fig


def build_genealogy_dot(state: EvolutionState) -> str:
    """Builds the genealogy graph in DOT format (Graphviz)."""

    def score_to_color(score: float | None) -> str:
        if score is None:
            return "#888888"
        if score >= 0.75:
            return "#27ae60"
        if score >= 0.50:
            return "#f39c12"
        return "#c0392b"

    lines = [
        "digraph G {",
        '    rankdir=TB;',
        '    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fontcolor="white"];',
        '    edge [color="#555555", arrowsize=0.7];',
    ]

    for genome in state.all_genomes:
        tag = genome.technique_tags[0] if genome.technique_tags else "?"
        color = score_to_color(genome.score)
        label = f"Gen {genome.generation} | {tag}\\nScore: {genome.score_display}"
        lines.append(f'    "{genome.id}" [label="{label}", fillcolor="{color}"];')

    for genome in state.all_genomes:
        for parent_id in genome.parent_ids:
            lines.append(f'    "{parent_id}" -> "{genome.id}";')

    lines.append("}")
    return "\n".join(lines)


def render_genome_card(genome: PromptGenome) -> None:
    """Displays an individual variant in a st.container."""
    score = genome.score or 0.0
    icon = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    tag = genome.technique_tags[0] if genome.technique_tags else "?"

    st.markdown(f"**{icon} {genome.score_display}** &nbsp; `{tag}`", unsafe_allow_html=True)

    if genome.rationale:
        st.caption(f"_{genome.rationale}_")

    with st.expander("View full prompt"):
        st.code(genome.prompt_text, language=None)

    if genome.response_sample:
        with st.expander("View produced response"):
            st.markdown(genome.response_sample)

    reasoning = genome.score_details.get("llm_reasoning", "")
    if reasoning and isinstance(reasoning, str):
        st.caption(f"💬 {reasoning[:120]}")


def render_generation_results(state: EvolutionState) -> None:
    """Displays all generations as expanders."""
    for result in reversed(state.generation_results):
        if result.generation == 0:
            label = f"🌱 Generation 0 — Initial prompt · Score: {result.max_score:.3f}"
        else:
            label = (
                f"🔬 Generation {result.generation} "
                f"· Max: {result.max_score:.3f} "
                f"· Mean: {result.mean_score:.3f}"
            )
        is_last = result.generation == len(state.generation_results) - 1

        with st.expander(label, expanded=is_last):
            sorted_genomes = sorted(
                result.genomes,
                key=lambda g: g.score or 0.0,
                reverse=True,
            )
            cols = st.columns(min(len(sorted_genomes), 3))
            for i, genome in enumerate(sorted_genomes):
                with cols[i % 3]:
                    render_genome_card(genome)


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("🧬 Prompt Evolution Engine")
    st.caption(
        "Automatically optimises a prompt via genetic algorithm — "
        "each generation produces higher-performing variants."
    )

    # ── Sidebar: configuration ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )

        st.divider()
        st.subheader("📋 Task")

        task_description = st.text_area(
            "Task description",
            value="Summarise a technical text in 3 clear and concise bullet points",
            height=80,
        )
        task_input = st.text_area(
            "Test input",
            value=(
                "Transformers use the attention mechanism to process token sequences "
                "in parallel. Unlike RNNs, they have no recurrence, "
                "which enables better parallelisation. The self-attention mechanism computes "
                "similarity scores between all pairs of tokens."
            ),
            height=130,
        )
        initial_prompt = st.text_area(
            "Initial prompt",
            value="Summarise this text in 3 bullet points.",
            height=80,
        )

        st.divider()
        st.subheader("🎯 Evaluation criteria")

        use_llm_judge = st.checkbox("LLM-as-judge", value=True)
        llm_criteria = ""
        if use_llm_judge:
            llm_criteria = st.text_area(
                "Criteria for the LLM judge",
                value=(
                    "The bullet points are clear, informative and cover the essential concepts. "
                    "They are formatted with dashes and do not exceed 2 lines each."
                ),
                height=100,
            )

        with st.expander("Deterministic criteria (optional)"):
            required_kw_raw = st.text_input(
                "Required keywords (comma-separated)",
                placeholder="transformer, attention",
            )
            require_json = st.checkbox("Response must be valid JSON", value=False)
            use_length = st.checkbox("Length constraint", value=False)
            length_range: tuple[int, int] | None = None
            if use_length:
                col1, col2 = st.columns(2)
                min_len = col1.number_input("Min (chars)", value=50, min_value=0)
                max_len = col2.number_input("Max (chars)", value=500, min_value=1)
                length_range = (int(min_len), int(max_len))

        st.divider()
        st.subheader("🔬 Evolution parameters")

        col1, col2 = st.columns(2)
        population_size = col1.number_input("Variants / gen", min_value=2, max_value=10, value=5)
        num_generations = col2.number_input("Generations", min_value=1, max_value=8, value=4)
        num_survivors = st.slider(
            "Survivors / gen",
            min_value=1,
            max_value=int(population_size),
            value=min(2, int(population_size)),
        )

        # Cost estimate
        judge_calls = int(population_size) * int(num_generations) if use_llm_judge else 0
        total_calls = 1 + int(num_generations) * int(population_size) + 1 + judge_calls
        estimated_cost = total_calls * 0.0003
        st.caption(f"~{total_calls} estimated API calls · **~{estimated_cost:.3f}€**")

        run_btn = st.button("🚀 Start evolution", type="primary", use_container_width=True)

    # ── Home page (before run) ────────────────────────────────────────
    if not run_btn:
        col1, col2, col3 = st.columns(3)
        col1.metric("Algorithm", "Genetic")
        col2.metric("Evaluator", "LLM-as-judge + Heuristics")
        col3.metric("Model", "GPT-4o-mini")

        st.markdown("---")
        st.markdown("""
        ### How it works

        | Step | Description |
        |------|-------------|
        | **Gen 0** | Your initial prompt is run and scored |
        | **Mutation** | An LLM generates N variants with different techniques (CoT, few-shot, persona…) |
        | **Scoring** | Each variant is run and evaluated (heuristics + LLM-judge) |
        | **Selection** | Tournament selection: the best survive |
        | **Repeat** | Up to K generations |

        **Output**: optimal prompt · evolution curve · genealogy tree
        """)
        return

    if not api_key:
        st.error("⚠️ Enter your OpenAI API key in the sidebar.")
        return

    # ── Build configuration objects ─────────────────────────────────────
    required_keywords = [kw.strip() for kw in required_kw_raw.split(",") if kw.strip()] \
        if required_kw_raw else []

    eval_criteria = EvalCriteria(
        llm_judge_description=llm_criteria if use_llm_judge else "",
        use_llm_judge=use_llm_judge,
        required_keywords=required_keywords,
        require_valid_json=require_json,
        target_length_range=length_range,
    )

    evolution_config = EvolutionConfig(
        task_description=task_description,
        task_input=task_input,
        initial_prompt=initial_prompt,
        eval_criteria=eval_criteria,
        population_size=int(population_size),
        num_generations=int(num_generations),
        num_survivors=int(num_survivors),
    )

    client = OpenAI(api_key=api_key)
    engine = EvolutionEngine(client)

    # ── Placeholders for live updates ─────────────────────────────────────
    progress_bar = st.progress(0.0, text="Initialising…")

    tab_evo, tab_best, tab_tree = st.tabs(["📈 Evolution", "🏆 Best Prompt", "🌳 Genealogy"])

    with tab_evo:
        chart_ph = st.empty()
        cards_ph = st.empty()

    with tab_best:
        best_ph = st.empty()

    with tab_tree:
        tree_ph = st.empty()

    # ── Evolution loop ────────────────────────────────────────────────
    try:
        for state in engine.run(evolution_config):
            total_gens = evolution_config.num_generations + 1
            progress = state.current_generation / total_gens
            gen_label = (
                f"Generation {state.current_generation - 1}/{evolution_config.num_generations} "
                f"— {'complete' if state.is_complete else 'in progress…'}"
            )
            progress_bar.progress(min(progress, 1.0), text=gen_label)

            # Chart
            with tab_evo:
                chart_ph.plotly_chart(build_score_chart(state), use_container_width=True)
                with cards_ph.container():
                    render_generation_results(state)

            # Best prompt
            best = state.best_genome
            if best:
                with tab_best:
                    with best_ph.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Best score", f"{best.score:.3f}" if best.score else "—")
                        col2.metric("Technique", best.technique_tags[0] if best.technique_tags else "—")
                        col3.metric("Generation", str(best.generation))
                        delta = state.improvement
                        col4.metric(
                            "Improvement",
                            f"+{delta:.3f}" if delta and delta > 0 else "—",
                            delta=delta,
                        )

                        st.markdown("#### Optimal prompt")
                        st.markdown(
                            f'<div class="best-prompt"><pre>{best.prompt_text}</pre></div>',
                            unsafe_allow_html=True,
                        )

                        if best.response_sample:
                            st.markdown("#### Produced response")
                            st.markdown(best.response_sample)

                        reasoning = best.score_details.get("llm_reasoning", "")
                        if reasoning and isinstance(reasoning, str):
                            st.info(f"💬 Evaluation: _{reasoning}_")

            # Genealogy
            with tab_tree:
                tree_ph.graphviz_chart(build_genealogy_dot(state), use_container_width=True)

    except Exception as error:
        st.error(f"Error during evolution: {error}")
        return

    progress_bar.progress(1.0, text="✅ Evolution complete!")
    st.balloons()


if __name__ == "__main__":
    main()

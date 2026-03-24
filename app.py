"""
Interface Streamlit du Prompt Evolution Engine.
Lance l'évolution et affiche les résultats en temps réel.
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

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prompt Evolution Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS custom ---
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


# ─── Helpers de visualisation ──────────────────────────────────────────────


def build_score_chart(state: EvolutionState) -> go.Figure:
    """Graphique Plotly de l'évolution des scores par génération."""
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
        mode="lines+markers", name="Moyenne",
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
        title={"text": "Score par génération", "font": {"size": 16}},
        xaxis_title="Génération",
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
    """Construit le graphe de généalogie au format DOT (Graphviz)."""

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
    """Affiche un variant individuel dans un st.container."""
    score = genome.score or 0.0
    icon = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    tag = genome.technique_tags[0] if genome.technique_tags else "?"

    st.markdown(f"**{icon} {genome.score_display}** &nbsp; `{tag}`", unsafe_allow_html=True)

    if genome.rationale:
        st.caption(f"_{genome.rationale}_")

    with st.expander("Voir le prompt complet"):
        st.code(genome.prompt_text, language=None)

    if genome.response_sample:
        with st.expander("Voir la réponse produite"):
            st.markdown(genome.response_sample)

    reasoning = genome.score_details.get("llm_reasoning", "")
    if reasoning and isinstance(reasoning, str):
        st.caption(f"💬 {reasoning[:120]}")


def render_generation_results(state: EvolutionState) -> None:
    """Affiche toutes les générations sous forme d'expanders."""
    for result in reversed(state.generation_results):
        if result.generation == 0:
            label = f"🌱 Génération 0 — Prompt initial · Score: {result.max_score:.3f}"
        else:
            label = (
                f"🔬 Génération {result.generation} "
                f"· Max: {result.max_score:.3f} "
                f"· Moy: {result.mean_score:.3f}"
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
        "Optimise automatiquement un prompt via algorithme génétique — "
        "chaque génération produit des variants plus performants."
    )

    # ── Sidebar : configuration ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )

        st.divider()
        st.subheader("📋 Tâche")

        task_description = st.text_area(
            "Description de la tâche",
            value="Résumer un texte technique en 3 bullet points clairs et concis",
            height=80,
        )
        task_input = st.text_area(
            "Entrée de test",
            value=(
                "Les transformers utilisent le mécanisme d'attention pour traiter les séquences "
                "de tokens en parallèle. Contrairement aux RNNs, ils n'ont pas de récurrence, "
                "ce qui permet une meilleure parallélisation. Le mécanisme self-attention calcule "
                "des scores de similarité entre tous les pairs de tokens."
            ),
            height=130,
        )
        initial_prompt = st.text_area(
            "Prompt initial",
            value="Résume ce texte en 3 bullet points.",
            height=80,
        )

        st.divider()
        st.subheader("🎯 Critères d'évaluation")

        use_llm_judge = st.checkbox("LLM-as-judge", value=True)
        llm_criteria = ""
        if use_llm_judge:
            llm_criteria = st.text_area(
                "Critères pour le juge LLM",
                value=(
                    "Les bullet points sont clairs, informatifs et couvrent les concepts essentiels. "
                    "Ils sont formatés avec des tirets et ne dépassent pas 2 lignes chacun."
                ),
                height=100,
            )

        with st.expander("Critères déterministes (optionnel)"):
            required_kw_raw = st.text_input(
                "Mots-clés requis (séparés par ',')",
                placeholder="transformer, attention",
            )
            require_json = st.checkbox("Réponse doit être du JSON valide", value=False)
            use_length = st.checkbox("Contrainte de longueur", value=False)
            length_range: tuple[int, int] | None = None
            if use_length:
                col1, col2 = st.columns(2)
                min_len = col1.number_input("Min (chars)", value=50, min_value=0)
                max_len = col2.number_input("Max (chars)", value=500, min_value=1)
                length_range = (int(min_len), int(max_len))

        st.divider()
        st.subheader("🔬 Paramètres d'évolution")

        col1, col2 = st.columns(2)
        population_size = col1.number_input("Variants / gen", min_value=2, max_value=10, value=5)
        num_generations = col2.number_input("Générations", min_value=1, max_value=8, value=4)
        num_survivors = st.slider(
            "Survivants / gen",
            min_value=1,
            max_value=int(population_size),
            value=min(2, int(population_size)),
        )

        # Estimation coût
        judge_calls = int(population_size) * int(num_generations) if use_llm_judge else 0
        total_calls = 1 + int(num_generations) * int(population_size) + 1 + judge_calls
        estimated_cost = total_calls * 0.0003
        st.caption(f"~{total_calls} appels API estimés · **~{estimated_cost:.3f}€**")

        run_btn = st.button("🚀 Lancer l'évolution", type="primary", use_container_width=True)

    # ── Page d'accueil (avant le run) ────────────────────────────────────
    if not run_btn:
        col1, col2, col3 = st.columns(3)
        col1.metric("Algorithme", "Génétique")
        col2.metric("Évaluateur", "LLM-as-judge + Heuristiques")
        col3.metric("Modèle", "GPT-4o-mini")

        st.markdown("---")
        st.markdown("""
        ### Comment ça marche ?

        | Étape | Description |
        |-------|-------------|
        | **Gen 0** | Ton prompt initial est exécuté et scoré |
        | **Mutation** | Un LLM génère N variants avec des techniques différentes (CoT, few-shot, persona…) |
        | **Scoring** | Chaque variant est exécuté et évalué (heuristiques + LLM-judge) |
        | **Sélection** | Tournament selection : les meilleurs survivent |
        | **Répète** | Jusqu'à K générations |

        **Résultat** : prompt optimal · courbe d'évolution · arbre de généalogie
        """)
        return

    if not api_key:
        st.error("⚠️ Renseigne ta clé API OpenAI dans la sidebar.")
        return

    # ── Construction des objets de configuration ─────────────────────────
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

    # ── Placeholders pour les mises à jour live ───────────────────────────
    progress_bar = st.progress(0.0, text="Initialisation…")

    tab_evo, tab_best, tab_tree = st.tabs(["📈 Évolution", "🏆 Meilleur Prompt", "🌳 Généalogie"])

    with tab_evo:
        chart_ph = st.empty()
        cards_ph = st.empty()

    with tab_best:
        best_ph = st.empty()

    with tab_tree:
        tree_ph = st.empty()

    # ── Boucle d'évolution ────────────────────────────────────────────────
    try:
        for state in engine.run(evolution_config):
            total_gens = evolution_config.num_generations + 1
            progress = state.current_generation / total_gens
            gen_label = (
                f"Génération {state.current_generation - 1}/{evolution_config.num_generations} "
                f"— {'terminé' if state.is_complete else 'en cours…'}"
            )
            progress_bar.progress(min(progress, 1.0), text=gen_label)

            # Graphique
            with tab_evo:
                chart_ph.plotly_chart(build_score_chart(state), use_container_width=True)
                with cards_ph.container():
                    render_generation_results(state)

            # Meilleur prompt
            best = state.best_genome
            if best:
                with tab_best:
                    with best_ph.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Meilleur score", f"{best.score:.3f}" if best.score else "—")
                        col2.metric("Technique", best.technique_tags[0] if best.technique_tags else "—")
                        col3.metric("Génération", str(best.generation))
                        delta = state.improvement
                        col4.metric(
                            "Amélioration",
                            f"+{delta:.3f}" if delta and delta > 0 else "—",
                            delta=delta,
                        )

                        st.markdown("#### Prompt optimal")
                        st.markdown(
                            f'<div class="best-prompt"><pre>{best.prompt_text}</pre></div>',
                            unsafe_allow_html=True,
                        )

                        if best.response_sample:
                            st.markdown("#### Réponse produite")
                            st.markdown(best.response_sample)

                        reasoning = best.score_details.get("llm_reasoning", "")
                        if reasoning and isinstance(reasoning, str):
                            st.info(f"💬 Évaluation : _{reasoning}_")

            # Généalogie
            with tab_tree:
                tree_ph.graphviz_chart(build_genealogy_dot(state), use_container_width=True)

    except Exception as error:
        st.error(f"Erreur durant l'évolution : {error}")
        return

    progress_bar.progress(1.0, text="✅ Évolution terminée !")
    st.balloons()


if __name__ == "__main__":
    main()

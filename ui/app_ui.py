"""Gradio dashboard. Layout only: all logic lives in ui.handlers.Handlers."""
from __future__ import annotations

import gradio as gr
import pandas as pd

from config import CATEGORIES
from .handlers import Handlers
from .theme import CSS


def build_ui(ctx) -> gr.Blocks:
    h = Handlers(ctx)
    s = ctx.settings
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    with gr.Blocks(theme=theme, css=CSS, title="ResearchMind AI") as demo:
        gr.HTML('<div id="rm-header"><h1>ResearchMind AI</h1><p>Intelligent research paper analysis and discovery platform '
                '<span class="ai-badge">AI-assisted: verify against cited pages</span></p></div>')
        doc_single: list[gr.Dropdown] = []
        doc_multi: list[gr.Dropdown] = []

        def refresh_dropdowns():
            ch = h.doc_choices()
            return [gr.update(choices=ch, value=None) for _ in doc_single + doc_multi]

        with gr.Tabs():
            # 1 Home
            with gr.Tab("Home"):
                status_md = gr.Markdown()
                cat_plot = gr.BarPlot(x="Category", y="Papers", title="Papers by predicted category", height=320)
                gr.Button("Refresh dashboard").click(h.dashboard, None, [status_md, cat_plot])
                demo.load(h.dashboard, None, [status_md, cat_plot])
            # 2 Library
            with gr.Tab("Document Library"):
                lib_df = gr.Dataframe(value=h.library_table, interactive=False, wrap=True, label="Indexed papers")
                with gr.Row():
                    lib_pick = gr.Dropdown(label="Paper", choices=h.doc_choices())
                    doc_single.append(lib_pick)
                    del_btn = gr.Button("Delete selected paper", variant="stop")
                    reidx_btn = gr.Button("Re-index all documents")
                lib_msg = gr.Markdown()
                gr.Button("Refresh").click(h.library_table, None, lib_df)
            # 3 Upload
            with gr.Tab("Upload"):
                gr.Markdown("Upload one or more text-based or scanned research papers (PDF). Duplicates are detected automatically.")
                files = gr.File(label="PDF files", file_count="multiple", file_types=[".pdf"])
                up_btn = gr.Button("Process and index", variant="primary")
                up_out = gr.Dataframe(label="Result per file", wrap=True, interactive=False)
            # 4 Summarization
            with gr.Tab("Summarization"):
                with gr.Row():
                    sum_doc = gr.Dropdown(label="Paper", choices=h.doc_choices())
                    doc_single.append(sum_doc)
                    sum_force = gr.Checkbox(label="Regenerate (ignore cache)")
                    sum_btn = gr.Button("Summarize", variant="primary")
                sum_md = gr.Markdown()
                sum_src = gr.Dataframe(label="Source passages ([n] markers above)", wrap=True, interactive=False)
            # 5 Ask
            with gr.Tab("Ask Your Papers"):
                q = gr.Textbox(label="Question", placeholder="e.g. Which datasets were used and what accuracy was reported?", lines=2)
                with gr.Row():
                    ask_docs = gr.Dropdown(label="Restrict to papers (empty = all)", choices=h.doc_choices(), multiselect=True)
                    doc_multi.append(ask_docs)
                    mode = gr.Radio(["hybrid", "vector", "bm25"], value="hybrid", label="Retrieval")
                with gr.Row():
                    topk = gr.Slider(1, 15, value=s.top_k, step=1, label="Top-K")
                    rerank = gr.Checkbox(value=True, label="Cross-encoder reranking")
                ask_btn = gr.Button("Ask", variant="primary")
                ans = gr.Markdown(label="Answer")
                cites = gr.Dataframe(label="Citations (document and page)", wrap=True, interactive=False)
                with gr.Accordion("Retrieved passages and scores", open=False):
                    ret = gr.Dataframe(wrap=True, interactive=False)
            # 6 Comparison
            with gr.Tab("Comparison"):
                cmp_docs = gr.Dropdown(label="Select two or more papers", choices=h.doc_choices(), multiselect=True)
                doc_multi.append(cmp_docs)
                cmp_btn = gr.Button("Compare", variant="primary")
                cmp_tbl = gr.Dataframe(label="Comparison table", wrap=True, interactive=False)
                cmp_md = gr.Markdown()
            # 7 Gaps
            with gr.Tab("Research Gaps"):
                gap_docs = gr.Dropdown(label="Select two or more papers", choices=h.doc_choices(), multiselect=True)
                doc_multi.append(gap_docs)
                gap_btn = gr.Button("Analyze", variant="primary")
                gap_md = gr.Markdown()
                with gr.Row():
                    gap_kw = gr.Dataframe(label="Distinctive terms per paper", wrap=True, interactive=False)
                    gap_m = gr.Dataframe(label="Method mentions (lexicon counts)", wrap=True, interactive=False)
            # 8 NLP & classification
            with gr.Tab("NLP and Classification"):
                nlp_doc = gr.Dropdown(label="Paper", choices=h.doc_choices())
                doc_single.append(nlp_doc)
                nlp_btn = gr.Button("Analyze", variant="primary")
                nlp_label = gr.Label(label=f"Predicted category (top 5 of {len(CATEGORIES)})")
                nlp_info = gr.Markdown()
                with gr.Row():
                    nlp_kw = gr.Dataframe(label="Keywords", interactive=False)
                    nlp_ent = gr.Dataframe(label="Scientific entities", interactive=False)
                with gr.Accordion("Classifier quality (cross-validation on bundled seed data)", open=False):
                    m_btn = gr.Button("Run cross-validation")
                    m_df = gr.Dataframe(interactive=False)
                    m_md = gr.Markdown()
            # 9 Evaluation
            with gr.Tab("RAG Evaluation"):
                gr.Markdown("Compares vector, hybrid and hybrid+rerank retrieval on the bundled **synthetic** QA set. All numbers are measured at run time.")
                gen_chk = gr.Checkbox(label="Also run LLM-based generation metrics (needs API key; makes many LLM calls)")
                ev_btn = gr.Button("Run evaluation", variant="primary")
                ev_tbl = gr.Dataframe(label="Retrieval metrics", interactive=False)
                ev_plot = gr.BarPlot(x="Metric", y="Value", color="Configuration", title="Retrieval metrics", height=340)
                ev_md = gr.Markdown()
            # 10 Settings
            with gr.Tab("Settings"):
                gr.Markdown("API keys are **never** entered here: set `GEMINI_API_KEY` as an environment variable / Space secret.")
                gr.JSON(value=s.public_dict(), label="Active configuration (read-only)")
                with gr.Row():
                    st_cs = gr.Slider(50, 600, value=s.chunk_size, step=10, label="Chunk size (words)")
                    st_ov = gr.Slider(0, 200, value=s.chunk_overlap, step=5, label="Chunk overlap (words)")
                    st_k = gr.Slider(1, 15, value=s.top_k, step=1, label="Default top-K")
                with gr.Row():
                    st_v = gr.Slider(0, 1, value=s.min_vector_score, step=0.01, label="Min cosine similarity")
                    st_r = gr.Slider(0, 1, value=s.min_rerank_score, step=0.01, label="Min rerank score")
                st_btn = gr.Button("Apply", variant="primary")
                st_msg = gr.Markdown()

        # ---- wiring
        dd_outputs = doc_single + doc_multi

        def do_upload(f, progress=gr.Progress()):
            table = h.upload(f, progress)
            return [table, h.library_table()] + refresh_dropdowns()

        up_btn.click(do_upload, files, [up_out, lib_df] + dd_outputs)
        del_btn.click(lambda d: (h.delete(d), h.library_table()), lib_pick, [lib_msg, lib_df]).then(refresh_dropdowns, None, dd_outputs)
        reidx_btn.click(h.reindex, None, lib_msg)
        sum_btn.click(h.summarize, [sum_doc, sum_force], [sum_md, sum_src])
        ask_btn.click(h.ask, [q, ask_docs, mode, topk, rerank], [ans, cites, ret])
        q.submit(h.ask, [q, ask_docs, mode, topk, rerank], [ans, cites, ret])
        cmp_btn.click(h.compare, cmp_docs, [cmp_tbl, cmp_md])
        gap_btn.click(h.gap_analysis, gap_docs, [gap_md, gap_kw, gap_m])
        nlp_btn.click(h.nlp_view, nlp_doc, [nlp_label, nlp_kw, nlp_ent, nlp_info])
        m_btn.click(h.classifier_metrics, None, [m_df, m_md])

        def do_eval(gen):
            df, note = h.run_eval(gen)
            long = df.melt(id_vars="Configuration", var_name="Metric", value_name="Value") if len(df) else pd.DataFrame(columns=["Configuration", "Metric", "Value"])
            return df, long, note

        ev_btn.click(do_eval, gen_chk, [ev_tbl, ev_plot, ev_md])
        st_btn.click(h.apply_settings, [st_cs, st_ov, st_k, st_v, st_r], st_msg)
        demo.load(refresh_dropdowns, None, dd_outputs)
    return demo

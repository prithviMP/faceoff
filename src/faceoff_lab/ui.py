"""Student workbench: load, inspect, retrieve, curate, compare. Real evidence only."""

from __future__ import annotations
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
import streamlit as st
import yaml
from faceoff_lab.cli import EXAMPLES
from faceoff_lab.project import load_project, config_path
from faceoff_lab.datasets import load_records, source_columns
from faceoff_lab.rag import RAGIndex
from faceoff_lab.llm import LLMSettings
from faceoff_lab.engine import run, compare_traces, evaluate
from faceoff_lab.curation import draft_bundle, mark_reviewed, write_page
from faceoff_lab.okf import validate_bundle, load_page
from faceoff_lab.importing import import_path

CSS = """<style>
.block-container{max-width:1440px;padding:3.5rem 2.5rem 3rem}h1{font-size:38px!important;letter-spacing:-.03em}h2{font-size:27px!important}
[data-testid=stAppViewContainer]{background:#fbfaf7;color:#172b32}.stAppDeployButton{display:none}
p,li{font-size:18px}.eyebrow{font-size:14px;color:#506267;letter-spacing:.12em;font-weight:700;text-transform:uppercase}
.note{font-size:18px;color:#506267}.source{border-left:3px solid #286b66;background:#fff;padding:18px 22px;margin:14px 0;overflow-wrap:anywhere;min-width:0}
.source pre{white-space:pre-wrap;font:19px/1.5 system-ui}.answer{font-size:27px;font-weight:600;line-height:1.4;padding:12px 0}.state{font-size:16px;font-weight:700;color:#286b66;text-transform:uppercase;letter-spacing:.05em}a{color:#166858}
</style>"""


def activate(path):
    load_project(path)
    st.session_state.project_file = str(config_path(path))
    st.session_state.results = {}
    st.session_state.preview = None
    st.session_state.index = None
    st.session_state.step = "1 · Inspect data"


def save(project):
    project.save(st.session_state.project_file)
    st.session_state.index = None
    st.session_state.preview = None
    st.session_state.results = {}


def index_for(project):
    records = load_records(project)
    fresh = RAGIndex(project, records)
    cached = st.session_state.get("index")
    if cached is None or cached.fingerprint != fresh.fingerprint:
        st.session_state.index = fresh
    else:
        cached.settings = project.rag
    return st.session_state.index


def welcome():
    st.subheader("Choose data to work with")
    st.write(
        "Start with a small example or bring your own files. Without an answer provider, the lab shows evidence and checks; it does not write an answer."
    )
    left, right = st.columns(2, gap="large")
    with left:
        with st.form("example"):
            example = st.selectbox("Example", ["policies", "support", "refunds", "iris"])
            st.caption(
                "Policies, tickets and refunds are synthetic. Iris is a public UCI dataset with attribution included."
            )
            destination = st.text_input("New working folder", value="workspace/my-example")
            if st.form_submit_button("Start example"):
                dest = Path(destination).expanduser().resolve()
                if dest.exists():
                    raise ValueError(
                        "Choose a new working folder, or open the existing project on the right."
                    )
                shutil.copytree(EXAMPLES / example, dest)
                activate(dest)
                st.rerun()
    with right:
        with st.form("open"):
            path = st.text_input("Existing project.yaml or project folder")
            if st.form_submit_button("Open project"):
                activate(path)
                st.rerun()
    st.divider()
    with st.expander("Import your own files", expanded=False):
        st.write(
            "Markdown, text, extractable PDF, CSV, JSONL, folders and ZIP files. Row mappings can be changed in the next step."
        )
        with st.form("import"):
            local = st.text_input("Local file or folder path")
            upload = st.file_uploader(
                "Or upload a supported file", type=["zip", "md", "txt", "pdf", "csv", "jsonl"]
            )
            dest = st.text_input("New project folder", value="workspace/my-import")
            if st.form_submit_button("Import dataset"):
                if upload:
                    data = upload.getvalue()
                    if len(data) > 100 * 1024 * 1024:
                        raise ValueError("Upload limit: 100 MB.")
                    staging = Path.cwd() / "workspace" / "upload-staging"
                    staging.mkdir(parents=True, exist_ok=True)
                    source = staging / (hashlib.sha256(data).hexdigest()[:12] + "-" + Path(upload.name).name)
                    source.write_bytes(data)
                elif local:
                    source = Path(local)
                else:
                    raise ValueError("Choose a local path or upload a file.")
                project = import_path(source, Path(dest).expanduser().resolve())
                activate(project.root)
                st.rerun()
        st.caption(
            "Imports stay on this machine. If you later enable a remote provider, selected evidence is sent to that endpoint. Check permission and license before importing or sharing a dataset."
        )


def data_step(project):
    st.subheader("What did you bring?")
    st.caption(str(project.root))
    for i, source in enumerate(project.sources):
        with st.expander(f"{source.kind} · {source.path}", expanded=source.kind != "documents"):
            if source.kind in {"csv", "jsonl"}:
                columns = source_columns(project.resolve(source.path), source.kind)
                with st.form(f"map-{i}"):
                    idcol = st.selectbox(
                        "Record ID column",
                        ["Use row numbers"] + columns,
                        index=(columns.index(source.id_column) + 1 if source.id_column in columns else 0),
                    )
                    textcols = st.multiselect(
                        "Text to search", columns, default=[c for c in source.text_columns if c in columns]
                    )
                    meta = st.multiselect(
                        "Metadata to keep (including date fields)",
                        columns,
                        default=[c for c in source.metadata_columns if c in columns],
                    )
                    if st.form_submit_button("Save column mapping"):
                        if not textcols:
                            raise ValueError("Select at least one text column.")
                        source.id_column = idcol if idcol != "Use row numbers" else "__row_number__"
                        source.text_columns = textcols
                        source.metadata_columns = meta
                        save(project)
                        st.rerun()
            else:
                st.write(
                    "Documents are loaded as text and split later. Scanned PDFs require OCR outside this lab."
                )
    records = load_records(project)
    total = sum(len(r.text.encode()) for r in records)
    a, b, c = st.columns(3)
    a.metric("Records", len(records))
    b.metric("Source files", len({r.source for r in records}))
    c.metric("Extracted text", f"{total:,} bytes")
    st.write(
        "Search can index these records automatically. A handbook needs selected topics, useful descriptions and explicit review."
    )
    sample = st.selectbox("Inspect a raw record", range(len(records)), format_func=lambda i: records[i].id)
    record = records[sample]
    st.code(record.source, language=None)
    st.text_area("Loaded text", record.text, height=220, disabled=True)
    st.json(record.metadata)


def search_step(project):
    st.subheader("Build and inspect search")
    with st.form("search-settings"):
        a, b, c = st.columns(3)
        size = a.number_input(
            "Characters per piece", min_value=50, max_value=10000, value=project.rag.chunk_size
        )
        overlap = b.number_input(
            "Overlapping characters", min_value=0, max_value=9999, value=project.rag.overlap
        )
        top = c.number_input("Pieces to give the model", min_value=1, max_value=50, value=project.rag.top_k)
        semantic = st.checkbox("Add meaning search (optional local model)", value=project.rag.semantic)
        model = st.text_input("Embedding model or local model folder", value=project.rag.embedding_model)
        rerank = st.checkbox("Recheck the best matches with a second model", value=project.rag.rerank)
        rankmodel = st.text_input("Second model or local folder", value=project.rag.reranker_model)
        if st.form_submit_button("Apply search settings"):
            project.rag.chunk_size = int(size)
            project.rag.overlap = int(overlap)
            project.rag.top_k = int(top)
            project.rag.semantic = semantic
            project.rag.embedding_model = model
            project.rag.rerank = rerank
            project.rag.reranker_model = rankmodel
            project.validate()
            save(project)
            st.rerun()
    index = index_for(project)
    st.write(f"{len(index.records):,} records → {len(index.chunks):,} searchable pieces")
    st.caption(
        "Keyword search: BM25+. Meaning search adds multilingual embeddings; the two rankings are combined. Scores rank matches; they do not measure truth."
    )
    if project.rag.semantic:
        st.info(
            "Meaning index ready."
            if index.embeddings_ready
            else "This content/settings combination has no prepared meaning index."
        )
        if st.button("Prepare semantic index"):
            status = st.empty()
            with st.spinner("Preparing the CPU index; the first use may download the selected model…"):
                index.prepare(lambda message: status.info(message))
            st.success("Meaning index prepared for this content.")
    question = st.text_input("Preview a search before generating an answer", key="preview_question")
    if st.button("Preview retrieval"):
        selected, trace = index.search(question)
        st.session_state.preview = (selected, trace)
    if st.session_state.get("preview"):
        selected, trace = st.session_state.preview
        st.dataframe(trace["candidates"], hide_index=True, width="stretch")
        for i, source in enumerate(selected, 1):
            passage(source, f"preview-{i}", i)
        if not selected:
            st.info("No matching passages. Try another question or inspect the text columns.")
        with st.expander("Complete search trace"):
            st.json(trace)


def handbook_step(project):
    st.subheader("Create or curate the handbook")
    st.write(
        "Handbook pages are a curated knowledge layer. Drafting copies source text; it does not verify it or invent approved calculations."
    )
    with st.form("bundle-path"):
        bundle = st.text_input(
            "Existing bundle folder (relative to project or absolute)", value=project.okf_root or ""
        )
        if st.form_submit_button("Use this bundle"):
            if not (project.resolve(bundle) / "index.md").is_file():
                raise ValueError("The bundle needs a root index.md.")
            project.okf_root = bundle
            save(project)
            st.rerun()
    with st.expander("Create an unverified draft from selected records"):
        records = load_records(project)
        chosen = st.multiselect("Records to draft", range(len(records)), format_func=lambda i: records[i].id)
        destination = st.text_input("New draft folder", value="handbook-draft")
        if st.button("Create draft handbook"):
            if not chosen:
                raise ValueError("Choose the records that belong in the handbook.")
            path = project.resolve(destination)
            draft_bundle([records[i] for i in chosen], path, project.root)
            project.okf_root = os.path.relpath(path, project.root)
            save(project)
            st.rerun()
    if not project.okf_path:
        return
    root = project.okf_path
    st.code(
        "\n".join("  " * len(p.relative_to(root).parts[:-1]) + p.name for p in sorted(root.rglob("*.md"))),
        language=None,
    )
    report = validate_bundle(root)
    (st.success if report["valid"] else st.error)(
        "Bundle structure is valid." if report["valid"] else "Fix bundle validation errors."
    )
    for issue in report["errors"]:
        st.error(issue)
    for warning in report["warnings"]:
        st.warning(warning)
    files = sorted(root.rglob("*.md"))
    if not files:
        return
    selected = st.selectbox("Open a handbook file", files, format_func=lambda p: str(p.relative_to(root)))
    page = load_page(selected)
    with st.form("edit-" + str(selected)):
        label = st.text_area("YAML label", yaml.safe_dump(page.label, sort_keys=False), height=180)
        body = st.text_area("Page or contents descriptions", page.body, height=240)
        if st.form_submit_button("Save this file"):
            parsed = yaml.safe_load(label) or {}
            if not isinstance(parsed, dict):
                raise ValueError("Metadata must be a YAML mapping.")
            write_page(selected, parsed, body)
            st.session_state.results = {}
            st.rerun()
    if selected.name not in {"index.md", "log.md"}:
        reviewer = st.text_input("Your reviewer ID", key="reviewer-" + str(selected))
        confirm = st.checkbox("I checked this page against its sources", key="confirm-" + str(selected))
        if st.button("Record my review"):
            if not confirm:
                raise ValueError("Review the sources and confirm before recording review.")
            mark_reviewed(selected, reviewer)
            st.session_state.results = {}
            st.rerun()
        st.caption(
            "Review entries are local declarations, not authenticated or cryptographic signatures. Review does not automatically change status or expiry."
        )
    with st.expander("Advanced · computations"):
        st.write(
            "An imported computation must declare SQLite runtime, a database path, typed parameters and a SQL fence. Only human-reviewed computations execute. Inspect the refunds example. Query connections are read-only; receipt checks run without an LLM. Do not approve an unfamiliar imported recipe without review."
        )


def passage(source, anchor, number):
    st.markdown(
        f'<div id="{html.escape(anchor)}" class="source"><strong>[{number}] {html.escape(source["source"])}</strong><div class="note">{html.escape(source["id"])}</div><pre>{html.escape(source["text"])}</pre></div>',
        unsafe_allow_html=True,
    )
    if "start" in source:
        st.caption(f"Exact record characters {source['start']}–{source['end']}; end is exclusive.")
    with st.expander(f"Source {number} metadata", expanded=False):
        st.json(source.get("metadata", {}))


def render_result(result):
    st.subheader(result["system"])
    st.markdown(
        f'<div class="state">{html.escape(result["status"].replace("_", " "))}</div>', unsafe_allow_html=True
    )
    if result["answer"]:
        answer = result["answer"]
        for citation in result["citations"]:
            n = citation["number"]
            answer = answer.replace(f"[{n}]", f"[{n}](#source-{result['run_id']}-{n})")
        st.markdown(answer)
    if result["reason"]:
        st.info(result["reason"])
    p = result["provider"]
    st.caption(
        f"{result['latency_seconds']:.3f} s · {result['words_read']:,} evidence words · {p['provider']} {p['model']}"
    )
    for warning in result.get("citation_warnings", []):
        st.warning(warning)
    for warning in result["trace"].get("warnings", []):
        st.warning(warning)
    st.write("Actual admitted evidence")
    for n, source in enumerate(result["sources"], 1):
        passage(source, f"source-{result['run_id']}-{n}", n)
    with st.expander("Complete reading / execution trace"):
        st.json(result["trace"])
    st.download_button(
        "Export this run",
        json.dumps(result, indent=2, default=str),
        file_name=f"{result['system']}-{result['run_id']}.json",
        key="export-" + result["run_id"],
    )


def compare_step(project):
    st.subheader("One question. Two evidence paths.")
    provider = st.selectbox(
        "Answer provider",
        ["none", "gemini", "ollama", "openai-compatible"],
        format_func=lambda s: "Retrieval only — no generated answer" if s == "none" else s,
    )
    model, url, temp = "", "", 0.0
    if provider != "none":
        with st.expander("Provider connection settings", expanded=True):
            model = st.text_input(
                "Model name", value=os.getenv("GEMINI_MODEL", "") if provider == "gemini" else ""
            )
            if provider in {"ollama", "openai-compatible"}:
                url = st.text_input(
                    "Endpoint (Ollama or OpenAI-compatible)",
                    value="http://localhost:11434" if provider == "ollama" else "",
                )
            temp = st.number_input("Temperature", 0.0, 2.0, 0.0, step=0.1)
    st.caption("Selected evidence goes to your chosen provider. Retrieval-only mode makes no model request.")
    question = st.text_input("Question", key="question")
    with st.expander("Optional · computation parameters and your evaluation key"):
        rawparams = st.text_area("Parameter values as JSON", value="{}")
        key = st.file_uploader("Your answer key (JSON; optional)", type=["json"])
        st.caption(
            'Example: {"your exact question": {"contains": ["phrase"], "status": "answered"}}. No built-in grading or winners.'
        )
    cols = st.columns(3)
    for col, label, systems in zip(
        cols, ["Run RAG", "Run OKF", "Run both"], [["RAG"], ["OKF"], ["RAG", "OKF"]]
    ):
        if col.button(label, width="stretch", type="primary" if label == "Run both" else "secondary"):
            params = json.loads(rawparams)
            if not isinstance(params, dict):
                raise ValueError("Parameters must be a JSON object.")
            settings = LLMSettings(provider=provider, model=model, base_url=url, temperature=temp)
            with st.spinner("Running the actual pipelines…"):
                index = index_for(project) if "RAG" in systems else None
                results = {s: run(project, s, question, settings, index, params) for s in systems}
            st.session_state.results = results
    results = st.session_state.get("results", {})
    if results:
        for col, result in zip(st.columns(len(results), gap="large"), results.values()):
            with col:
                render_result(result)
                if key:
                    expectations = json.loads(key.getvalue())
                    st.write("Your evaluation:", evaluate(result, expectations, result["question"]))
        st.write("Why the paths differ — observations from these traces")
        for observation in compare_traces(results):
            st.write("• " + observation)
        st.caption(
            "These cards are saved run snapshots. Changed question/provider controls apply to the next run."
        )


def main():
    st.set_page_config(page_title="Faceoff Lab", page_icon="↔", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Faceoff Lab / build · inspect · compare</div>', unsafe_allow_html=True)
    st.title("Faceoff Lab")
    if not st.session_state.get("project_file"):
        try:
            welcome()
        except Exception as exc:
            st.error(str(exc))
        return
    project = load_project(st.session_state.project_file)
    with st.sidebar:
        st.subheader(project.name)
        st.caption(project.description)
        st.code(str(project.root), language=None)
        if st.button("Change dataset"):
            st.session_state.project_file = None
            st.session_state.results = {}
            st.rerun()
        st.download_button(
            "Export project settings",
            yaml.safe_dump(project.to_dict(), sort_keys=False),
            file_name="project.yaml",
        )
        st.caption(
            "Source paths are relative to this project where possible. Share source files alongside these settings; credentials are never exported."
        )
    step = st.radio(
        "Your experiment",
        ["1 · Inspect data", "2 · Build RAG", "3 · Curate OKF", "4 · Compare"],
        horizontal=True,
        key="step",
    )
    try:
        {
            "1 · Inspect data": data_step,
            "2 · Build RAG": search_step,
            "3 · Curate OKF": handbook_step,
            "4 · Compare": compare_step,
        }[step](project)
    except Exception as exc:
        st.error(str(exc))
    st.divider()
    st.caption(
        "OKF is a knowledge format. This project supplies the reader, checks and local computation runtime. Compare the evidence and checks for your own dataset."
    )


if __name__ == "__main__":
    main()

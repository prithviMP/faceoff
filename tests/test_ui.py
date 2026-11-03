from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from faceoff_lab.project import ProjectConfig, SourceConfig
from faceoff_lab.curation import draft_bundle
from faceoff_lab.datasets import load_records

UI = Path(__file__).resolve().parents[1] / "src/faceoff_lab/ui.py"


def button(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception


def field(app, label, value):
    next(f for f in app.text_input if f.label == label).set_value(value)


def step(app, name):
    next(r for r in app.radio if r.label == "Your experiment").set_value(name).run()
    assert not app.exception


def test_welcome_and_folder_import(tmp_path):
    data = tmp_path / "input"
    data.mkdir()
    (data / "research.txt").write_text("Greenhouse irrigation runs at dawn.")
    app = AppTest.from_file(str(UI), default_timeout=90).run()
    assert not app.exception
    field(app, "Local file or folder path", str(data))
    field(app, "New project folder", str(tmp_path / "project"))
    button(app, "Import dataset")
    assert app.session_state["project_file"] == str(tmp_path / "project/project.yaml")
    assert app.metric[0].value == "1"
    # Reopen through the project form to exercise a separate, clean session.
    app = AppTest.from_file(str(UI), default_timeout=90).run()
    field(app, "Existing project.yaml or project folder", str(tmp_path / "project"))
    button(app, "Open project")
    step(app, "2 · Build RAG")
    field(app, "Preview a search before generating an answer", "greenhouse irrigation")
    button(app, "Preview retrieval")
    assert app.session_state["preview"][0]
    step(app, "4 · Compare")
    field(app, "Question", "greenhouse irrigation")
    button(app, "Run both")
    results = app.session_state["results"]
    assert results["RAG"]["status"] == "evidence"
    assert results["OKF"]["status"] == "no_coverage"
    assert not results["RAG"]["answer"]


@pytest.mark.parametrize(
    "topic,question",
    [
        ("Cobalt cells power a lunar battery.", "lunar battery"),
        ("Botanical greenhouse irrigation uses rainwater.", "greenhouse irrigation"),
    ],
)
def test_custom_projects_comparison_and_metadata_editing(tmp_path, topic, question):
    (tmp_path / "source.txt").write_text(topic)
    p = ProjectConfig("Independent", [SourceConfig("documents", "source.txt")], root=tmp_path)
    draft_bundle(load_records(p), tmp_path / "handbook")
    (tmp_path / "handbook/index.md").write_text(f"# Contents\n[Reference](page-0001.md) — {question}.\n")
    p.okf_root = "handbook"
    p.save(tmp_path / "project.yaml")
    app = AppTest.from_file(str(UI), default_timeout=90).run()
    field(app, "Existing project.yaml or project folder", str(tmp_path))
    button(app, "Open project")
    step(app, "3 · Curate OKF")
    assert any("valid" in m.value for m in app.success)
    assert any("unverified" in w.value for w in app.warning)
    step(app, "4 · Compare")
    field(app, "Question", question)
    button(app, "Run both")
    result = app.session_state["results"]
    assert result["RAG"]["status"] == "evidence" and result["OKF"]["status"] == "evidence"
    assert result["OKF"]["trace"]["checks"]["trust"] == "unverified"
    assert not result["RAG"]["answer"] and not result["OKF"]["answer"]

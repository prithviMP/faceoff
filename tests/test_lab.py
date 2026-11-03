from datetime import datetime, timezone
import json
import shutil
import sqlite3
import zipfile
import pytest
from faceoff_lab.project import ProjectConfig, SourceConfig, load_project
from faceoff_lab.datasets import load_records, register_loader, SourceRecord
from faceoff_lab.rag import RAGIndex
from faceoff_lab.engine import run, evaluate
from faceoff_lab.llm import LLMSettings
from faceoff_lab.okf import validate_bundle, load_page, check_label
from faceoff_lab.curation import draft_bundle, write_page, mark_reviewed
from faceoff_lab.importing import import_path, unpack_zip
from faceoff_lab.receipts import execute, inspect
from faceoff_lab.cli import EXAMPLES


def project(tmp_path, text="The library opens at nine."):
    (tmp_path / "notes.txt").write_text(text)
    return ProjectConfig("Custom", [SourceConfig("documents", "notes.txt")], root=tmp_path)


@pytest.mark.parametrize("name,expected", [("policies", 2), ("support", 30), ("refunds", 1), ("iris", 150)])
def test_packaged_examples_without_legacy_data(tmp_path, name, expected):
    shutil.copytree(EXAMPLES / name, tmp_path / name)
    config = load_project(tmp_path / name)
    assert len(load_records(config)) == expected
    assert not config.rag.semantic
    result = run(
        config,
        "RAG",
        "refunds" if name == "refunds" else "Iris-setosa" if name == "iris" else "mixer grinder",
    )
    assert result["status"] == "evidence"
    assert result["answer"] is None
    assert result["sources"]


def test_no_match_and_optional_evaluation(tmp_path):
    p = project(tmp_path)
    result = run(p, "RAG", "volcano")
    assert result["status"] == "insufficient_evidence" and not result["answer"]
    assert evaluate(result, {}, "volcano")["status"] == "ungraded"
    assert evaluate(result, {"volcano": {"status": "insufficient_evidence"}}, "volcano")["status"] == "pass"


@pytest.mark.parametrize(
    "topic,question",
    [
        ("The botanical greenhouse opens at ten.", "greenhouse opens"),
        ("The lunar battery uses cobalt cells.", "lunar battery"),
    ],
)
def test_unrelated_custom_import_index_query_okf_flow(tmp_path, topic, question):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "reference.txt").write_text(topic)
    p = import_path(incoming, tmp_path / "project")
    records = load_records(p)
    index = RAGIndex(p, records)
    result = run(p, "RAG", question, index=index)
    assert result["status"] == "evidence" and result["answer"] is None
    for source in result["sources"]:
        assert records[0].text[source["start"] : source["end"]] == source["text"]
    bundle = p.root / "handbook"
    draft_bundle(records, bundle, p.root)
    draft = load_page(bundle / "page-0001.md")
    assert (bundle / draft.label["sources"][0]["resource"]).is_file()
    write_page(
        bundle / "page-0001.md",
        {"type": "Reference", "title": "Reference", "status": "draft", "verified": []},
        topic,
    )
    (bundle / "index.md").write_text(f"# Contents\n[Reference](page-0001.md) — {question}.\n")
    p.okf_root = "handbook"
    p.save(p.root / "project.yaml")
    okf = run(load_project(p.root), "OKF", question)
    assert okf["status"] == "evidence"
    assert okf["trace"]["checks"]["trust"] == "unverified"
    assert okf["trace"]["warnings"]
    mark_reviewed(bundle / "page-0001.md", "student")
    assert load_page(bundle / "page-0001.md").label["verified"][0]["by"] == "human:student"
    # Reviewing does not silently change draft status.
    assert load_page(bundle / "page-0001.md").label["status"] == "draft"


def test_csv_jsonl_mapping_duplicate_ids_and_custom_adapter(tmp_path):
    (tmp_path / "measure.csv").write_text("key,observation,date\na,glacier is retreating,2020\n")
    (tmp_path / "journal.jsonl").write_text(
        json.dumps({"key": "b", "observation": "reef is growing", "date": "2021"}) + "\n"
    )
    p = ProjectConfig(
        "Research",
        [
            SourceConfig("csv", "measure.csv", id_column="key", text_columns=["observation"]),
            SourceConfig("jsonl", "journal.jsonl", id_column="key", text_columns=["observation"]),
        ],
        root=tmp_path,
    )
    records = load_records(p)
    assert [r.id for r in records] == ["a", "b"]
    assert records[0].metadata["date"] == "2020"
    p.sources[1].text_columns = ["absent"]
    with pytest.raises(ValueError, match="missing configured"):
        load_records(p)
    register_loader(
        "unit-test-adapter", lambda p, s: [SourceRecord("custom", "adapter", "custom evidence", "custom")]
    )
    p.sources = [SourceConfig("unit-test-adapter", "ignored")]
    assert load_records(p)[0].text == "custom evidence"


def test_bundle_validation_and_refusal_states(tmp_path):
    p = project(tmp_path)
    p.okf_root = "handbook"
    bundle = tmp_path / "handbook"
    bundle.mkdir()
    (bundle / "index.md").write_text("# Contents\n[Library](missing.md) — library opening hours.\n")
    assert not validate_bundle(bundle)["valid"]
    assert "broken link" in " ".join(validate_bundle(bundle)["errors"])
    for label, reason in [
        ({"type": "Policy", "status": "deprecated"}, "status"),
        ({"type": "Policy", "stale_after": "2000-01-01T00:00:00Z"}, "deadline"),
    ]:
        write_page(bundle / "policy.md", label, "The library opens at nine.")
        (bundle / "index.md").write_text("# Contents\n[Library](policy.md) — library opening hours.\n")
        result = run(p, "OKF", "library opening hours")
        assert result["status"] in {"deprecated", "expired"} and reason in result["reason"]
        assert not result["sources"]
    write_page(bundle / "policy.md", {"type": "Policy", "stale_after": "not-a-date"}, "bad")
    assert not validate_bundle(bundle)["valid"]
    write_page(bundle / "policy.md", {}, "missing metadata")
    assert not validate_bundle(bundle)["valid"]


def test_real_receipts_read_only_and_tampering(tmp_path):
    db = tmp_path / "data.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE amounts(value INTEGER)")
        c.executemany("INSERT INTO amounts VALUES(?)", [(3,), (5,)])
    sql = "SELECT SUM(value) FROM amounts"
    receipt = execute(db, tmp_path / "receipts.sqlite", sql, {})
    assert receipt["result"] == 8
    assert inspect(receipt, sql, {})["passed"]
    for altered in [
        {**receipt, "result": 9},
        {**receipt, "sql_hash": "bad"},
        {**receipt, "executed_sql": "SELECT 9"},
    ]:
        assert not inspect(altered, sql, {})["passed"]
    changed = execute(db, tmp_path / "receipts.sqlite", "SELECT MAX(value) FROM amounts", {})
    check = inspect(changed, sql, {})
    assert not check["passed"] and "failed" in check["reason"]
    with pytest.raises((ValueError, sqlite3.OperationalError)):
        execute(db, tmp_path / "receipts.sqlite", "DELETE FROM amounts", {})
    # More than one returned row is kept; it is not silently reduced to the first.
    assert execute(db, tmp_path / "receipts.sqlite", "SELECT value FROM amounts", {})["result"] == [[3], [5]]


def test_computation_requires_review_and_parameters(tmp_path):
    shutil.copytree(EXAMPLES / "refunds", tmp_path / "refunds")
    p = load_project(tmp_path / "refunds")
    missing = run(p, "OKF", "refund total")
    assert missing["status"] == "needs_input"
    answer = run(p, "OKF", "refund total", parameters={"sale_id": "DEMO"})
    assert answer["status"] == "answered" and answer["trace"]["inspector"]["passed"]
    page = p.okf_path / "refunds.md"
    loaded = load_page(page)
    loaded.label["verified"] = []
    write_page(page, loaded.label, loaded.body)
    assert run(p, "OKF", "refund total", parameters={"sale_id": "DEMO"})["status"] == "unverified"


def test_citations_and_provider_errors_are_real_states(tmp_path, monkeypatch):
    from faceoff_lab import engine

    p = project(tmp_path)

    class TestProvider:
        def generate(self, prompt):
            return "Nine [1]. Unsupported citation [99]."

    monkeypatch.setattr(engine, "create_generator", lambda settings: TestProvider())
    result = run(p, "RAG", "library", LLMSettings(provider="ollama", model="test-fixture"))
    assert result["status"] == "answered"
    assert result["citations"][0]["source_id"] == result["sources"][0]["id"]
    assert result["citation_warnings"] == ["Unknown citation [99]"]

    class Failure:
        def generate(self, prompt):
            raise RuntimeError("Endpoint unavailable")

    monkeypatch.setattr(engine, "create_generator", lambda settings: Failure())
    result = run(p, "RAG", "library", LLMSettings(provider="ollama", model="test-fixture"))
    assert result["status"] == "error" and result["answer"] is None
    assert "unavailable" in result["reason"]


def test_zip_import_and_traversal_rejection(tmp_path):
    z = tmp_path / "input.zip"
    with zipfile.ZipFile(z, "w") as handle:
        handle.writestr("notes.txt", "Satellite antenna directions.")
    p = import_path(z, tmp_path / "imported")
    assert load_records(p)[0].text == "Satellite antenna directions."
    with zipfile.ZipFile(z, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        unpack_zip(z, tmp_path / "unsafe")
    assert not (tmp_path / "escape.txt").exists()


def test_index_identity_tracks_content_not_top_k(tmp_path):
    p = project(tmp_path)
    first = RAGIndex(p, load_records(p)).fingerprint
    p.rag.top_k = 9
    assert RAGIndex(p, load_records(p)).fingerprint == first
    (tmp_path / "notes.txt").write_text("Updated library hours.")
    assert RAGIndex(p, load_records(p)).fingerprint != first


def test_expiry_boundary():
    assert check_label({"stale_after": "2030-01-01T00:00:00Z"}, datetime(2029, 12, 31, tzinfo=timezone.utc))[
        "valid"
    ]
    assert not check_label(
        {"stale_after": "2030-01-01T00:00:00Z"}, datetime(2030, 1, 1, tzinfo=timezone.utc)
    )["valid"]


def test_extractable_pdf_loader(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 20 100 Td (Orbital beacon is blue.) Tj ET")
    page[NameObject("/Contents")] = content
    with (tmp_path / "sample.pdf").open("wb") as f:
        writer.write(f)
    p = ProjectConfig("Space", [SourceConfig("documents", "sample.pdf")], root=tmp_path)
    assert "Orbital beacon is blue." in load_records(p)[0].text


def test_actual_http_provider_adapters():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from threading import Thread
    from faceoff_lab.llm import OllamaGenerator, OpenAICompatibleGenerator

    received = []

    class FixtureServer(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            payload = (
                {"response": "test HTTP fixture"}
                if self.path == "/api/generate"
                else {"choices": [{"message": {"content": "test HTTP fixture"}}]}
            )
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), FixtureServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert (
            OllamaGenerator(LLMSettings(provider="ollama", model="fixture", base_url=base)).generate(
                "evidence"
            )
            == "test HTTP fixture"
        )
        assert (
            OpenAICompatibleGenerator(
                LLMSettings(provider="openai-compatible", model="fixture", base_url=base + "/v1")
            ).generate("evidence")
            == "test HTTP fixture"
        )
        assert received[0][0] == "/api/generate" and received[0][1]["model"] == "fixture"
        assert (
            received[1][0] == "/v1/chat/completions"
            and received[1][1]["messages"][0]["content"] == "evidence"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_failed_import_is_atomic_and_recursive_destination_is_rejected(tmp_path):
    invalid = tmp_path / "bad.zip"
    with zipfile.ZipFile(invalid, "w") as z:
        z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError):
        import_path(invalid, tmp_path / "new-project")
    assert not (tmp_path / "new-project").exists()
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.txt").write_text("Searchable notes")
    with pytest.raises(ValueError, match="outside"):
        import_path(source, source / "project")


def test_okf_root_relative_link_with_fragment(tmp_path):
    p = project(tmp_path)
    p.okf_root = "handbook"
    bundle = tmp_path / "handbook"
    bundle.mkdir()
    (bundle / "index.md").write_text(
        "# Contents\n[Library](/policies/library.md#hours) — library opening hours.\n"
    )
    write_page(bundle / "policies/library.md", {"type": "Reference"}, "The library opens at nine.")
    assert validate_bundle(bundle)["valid"]
    result = run(p, "OKF", "library opening hours")
    assert result["status"] == "evidence"
    assert result["sources"][0]["source"] == "policies/library.md"


def test_provider_can_declare_insufficient_evidence(tmp_path, monkeypatch):
    from faceoff_lab import engine

    class InsufficientProvider:
        def generate(self, prompt):
            return "INSUFFICIENT_EVIDENCE: the document has no closing time."

    monkeypatch.setattr(engine, "create_generator", lambda settings: InsufficientProvider())
    p = project(tmp_path)
    result = run(p, "RAG", "library closing time", LLMSettings(provider="ollama", model="fixture"))
    assert result["status"] == "insufficient_evidence"
    assert result["answer"] is None and "closing time" in result["reason"]
    assert result["sources"]

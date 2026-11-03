"""Cross-platform command-line checkpoints, using the same engines as the UI."""

from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from .project import load_project, config_path
from .datasets import load_records
from .rag import RAGIndex
from .llm import LLMSettings
from .engine import run
from .curation import draft_bundle
from .importing import import_path
from .okf import validate_bundle

EXAMPLES = Path(__file__).parent / "examples"


def main():
    parser = argparse.ArgumentParser(
        prog="faceoff", description="Inspect retrieval and handbook knowledge systems."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("ui")
    ui.add_argument("--port", type=int, default=8503)
    ex = sub.add_parser("example")
    ex.add_argument("name", choices=["policies", "support", "refunds", "iris"])
    ex.add_argument("destination", type=Path)
    imp = sub.add_parser("import")
    imp.add_argument("source", type=Path)
    imp.add_argument("destination", type=Path)
    for command in ["inspect", "index", "draft", "validate", "query"]:
        p = sub.add_parser(command)
        p.add_argument("project", type=Path)
        if command == "draft":
            p.add_argument("--destination", default="handbook-draft")
        if command == "query":
            p.add_argument("question")
            p.add_argument("--system", choices=["RAG", "OKF", "both"], default="both")
            p.add_argument(
                "--provider", choices=["none", "gemini", "ollama", "openai-compatible"], default="none"
            )
            p.add_argument("--model", default="")
            p.add_argument("--base-url", default="")
            p.add_argument("--parameters", default="{}", help="JSON values for a configured computation.")
    gen = sub.add_parser("generate-tickets")
    gen.add_argument("destination", type=Path)
    gen.add_argument("--rows", type=int, default=10000)
    args = parser.parse_args()
    try:
        if args.command == "ui":
            raise SystemExit(
                subprocess.call(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        str(Path(__file__).with_name("ui.py")),
                        "--server.port",
                        str(args.port),
                        "--server.address",
                        "127.0.0.1",
                        "--server.headless",
                        "true",
                    ]
                )
            )
        if args.command == "example":
            if args.destination.exists():
                raise ValueError("Destination already exists; choose a new folder.")
            shutil.copytree(EXAMPLES / args.name, args.destination)
            print(args.destination / "project.yaml")
            return
        if args.command == "import":
            project = import_path(args.source, args.destination)
            print(project.root / "project.yaml")
            return
        if args.command == "generate-tickets":
            if not 1 <= args.rows <= 100000:
                raise ValueError("Choose between 1 and 100,000 rows.")
            with args.destination.open("x", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "text", "language"])
                for i in range(args.rows):
                    w.writerow(
                        [
                            i + 1,
                            ["My refund is late.", "Mixer grinder toot gaya.", "Please update my address."][
                                i % 3
                            ],
                            ["English", "Hinglish", "English"][i % 3],
                        ]
                    )
            print(args.destination)
            return
        project = load_project(args.project)
        if args.command == "validate":
            if not project.okf_path:
                raise ValueError("Configure an OKF bundle first.")
            result = validate_bundle(project.okf_path)
            print(json.dumps(result, indent=2))
            raise SystemExit(0 if result["valid"] else 1)
        records = load_records(project)
        if args.command == "inspect":
            print(
                json.dumps(
                    {
                        "records": len(records),
                        "sources": sorted({r.source for r in records}),
                        "sample": [r.to_dict() for r in records[:3]],
                    },
                    indent=2,
                    default=str,
                )
            )
            return
        if args.command == "draft":
            destination = project.resolve(args.destination)
            draft_bundle(records, destination, project.root)
            project.okf_root = str(destination.relative_to(project.root))
            project.save(config_path(args.project))
            print(destination)
            return
        index = RAGIndex(project, records)
        if args.command == "index":
            if project.rag.semantic:
                index.prepare(print)
            print(
                json.dumps(
                    {
                        "chunks": len(index.chunks),
                        "fingerprint": index.fingerprint,
                        "semantic": project.rag.semantic,
                    }
                )
            )
            return
        systems = ["RAG", "OKF"] if args.system == "both" else [args.system]
        parameters = json.loads(args.parameters)
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a JSON object.")
        failed = False
        for system in systems:
            result = run(
                project,
                system,
                args.question,
                LLMSettings(provider=args.provider, model=args.model, base_url=args.base_url),
                index,
                parameters,
            )
            print(json.dumps(result, indent=2, default=str))
            failed = failed or result["status"] == "error"
        if failed:
            raise SystemExit(1)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

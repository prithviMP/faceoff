"""Read-only SQLite execution with independently inspectable local receipts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
import time
from pathlib import Path
from typing import Any


def digest(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def validate_read_only_sql(sql: str) -> None:
    stripped = sql.strip()
    if not re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise ValueError("Only SELECT queries (including SELECT with WITH) are allowed.")
    if ";" in stripped.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")


def execute(
    database: Path,
    receipt_database: Path,
    sql: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Execute one approved query without modifying the source database."""
    validate_read_only_sql(sql)
    if not database.is_file():
        raise FileNotFoundError(f"SQLite database not found: {database}")
    run_id = uuid.uuid4().hex
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        deadline = time.monotonic() + 10
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        rows = connection.execute(sql, parameters).fetchmany(1001)
        if len(rows) > 1000:
            raise ValueError("Computation exceeded the 1,000-row result limit.")
    result = rows[0][0] if len(rows) == 1 and len(rows[0]) == 1 else [list(row) for row in rows]
    receipt_database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(receipt_database) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS execution_receipts("
            "run_id TEXT PRIMARY KEY, database_path TEXT NOT NULL, executed_sql TEXT NOT NULL, "
            "parameters TEXT NOT NULL, result_json TEXT NOT NULL, sql_hash TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO execution_receipts VALUES(?,?,?,?,?,?)",
            (
                run_id,
                str(database.resolve()),
                sql,
                json.dumps(parameters, sort_keys=True),
                json.dumps(result),
                digest(sql),
            ),
        )
    return {
        "run_id": run_id,
        "database": str(database.resolve()),
        "executed_sql": sql,
        "parameters": parameters,
        "result": result,
        "sql_hash": digest(sql),
        "receipt_database": str(receipt_database),
    }


def inspect(
    receipt: dict[str, Any], approved_sql: str, expected_parameters: dict[str, Any]
) -> dict[str, Any]:
    """Reload the authoritative local receipt and compare every material field."""
    with sqlite3.connect(receipt["receipt_database"]) as connection:
        row = connection.execute(
            "SELECT database_path,executed_sql,parameters,result_json,sql_hash "
            "FROM execution_receipts WHERE run_id=?",
            (receipt.get("run_id"),),
        ).fetchone()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "receipt_exists",
        row is not None,
        "Execution record was found." if row else "No execution record exists.",
    )
    if row is None:
        return {"passed": False, "reason": "No authoritative execution record exists.", "checks": checks}
    database, actual_sql, parameters_json, result_json, stored_hash = row
    check("approved_sql", actual_sql == approved_sql, "Executed SQL matches the approved recipe.")
    check(
        "sql_fingerprint",
        receipt.get("sql_hash") == stored_hash == digest(actual_sql) == digest(approved_sql),
        "SQL fingerprints match.",
    )
    check("receipt_sql", receipt.get("executed_sql") == actual_sql, "Returned receipt SQL was not altered.")
    check("source_database", receipt.get("database") == database, "Source database matches the receipt.")
    check(
        "parameters",
        receipt.get("parameters") == json.loads(parameters_json) == expected_parameters,
        "Bound parameters match the approved execution.",
    )
    check(
        "result",
        receipt.get("result") == json.loads(result_json),
        "Displayed result matches the execution record.",
    )
    failed = next((item for item in checks if not item["passed"]), None)
    return {
        "passed": failed is None,
        "reason": "Recipe, parameters, source database, and result all match."
        if failed is None
        else f"Verification failed: {failed['name']}.",
        "approved_hash": digest(approved_sql),
        "checks": checks,
    }

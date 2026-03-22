from __future__ import annotations

import json
import os
from typing import Iterable

import pandas as pd

from .config import WRDS_CREDENTIALS_JSON


class WrdsUnavailableError(RuntimeError):
    pass


def load_wrds_credentials() -> dict[str, str]:
    if WRDS_CREDENTIALS_JSON.exists():
        payload = json.loads(WRDS_CREDENTIALS_JSON.read_text(encoding="utf-8"))
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", "")).strip()
        if username:
            return {"username": username, "password": password}
    username = str(os.getenv("WRDS_USERNAME", "")).strip()
    password = str(os.getenv("WRDS_PASSWORD", "")).strip()
    if username:
        return {"username": username, "password": password}
    return {"username": "", "password": ""}


def connect_wrds():
    try:
        import wrds
    except ImportError as exc:
        raise WrdsUnavailableError(
            "The 'wrds' package is not installed. Install dependencies from requirements.txt or place raw parquet files in data/raw/."
        ) from exc

    credentials = load_wrds_credentials()
    username = credentials["username"]
    password = credentials["password"]
    try:
        if username and password:
            return wrds.Connection(wrds_username=username, wrds_password=password)
        if username:
            return wrds.Connection(wrds_username=username)
        return wrds.Connection()
    except Exception as exc:  # pragma: no cover - depends on local WRDS environment
        raise WrdsUnavailableError(
            "Could not establish a WRDS connection. Set WRDS credentials or provide the expected raw parquet files."
        ) from exc


def execute_candidate_queries(connection, sql_candidates: Iterable[str]) -> pd.DataFrame:
    last_error: Exception | None = None
    for sql in sql_candidates:
        try:
            return connection.raw_sql(sql)
        except Exception as exc:  # pragma: no cover - depends on WRDS schema
            last_error = exc
    if last_error is None:
        raise RuntimeError("No SQL candidates were provided.")
    raise last_error


def test_wrds_connection() -> dict[str, object]:
    connection = connect_wrds()
    credentials = load_wrds_credentials()
    username = credentials["username"] or "interactive"
    current_user = connection.raw_sql("select current_user as current_user")

    libraries = sorted(str(name) for name in connection.list_libraries())
    visible_libraries = [name for name in ("crsp", "optionm", "optionm_all") if name in libraries]

    optionm_table_names: list[str] = []
    for library_name in ("optionm", "optionm_all"):
        if library_name not in libraries:
            continue
        try:
            table_names = {str(name) for name in connection.list_tables(library=library_name)}
        except Exception:  # pragma: no cover - depends on WRDS permissions/schema
            continue
        required_matches = sorted(
            name
            for name in table_names
            if name == "secnmd" or name == "secprd" or name.startswith("opprcd")
        )
        optionm_table_names.extend(f"{library_name}.{name}" for name in required_matches[:10])

    return {
        "connected": True,
        "credential_source": "json_file" if WRDS_CREDENTIALS_JSON.exists() else ("environment" if credentials["username"] else "interactive"),
        "requested_username": username,
        "database_user": str(current_user.iloc[0]["current_user"]),
        "available_libraries": visible_libraries,
        "wrds_library_count": len(libraries),
        "optionm_required_tables": optionm_table_names,
    }

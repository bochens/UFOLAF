from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufolaf_models import CountsTable, SampleMetadata, TemperatureFrozenFractionTable


SAMPLE_VALUE_RE = re.compile(r"^(?P<sample>.+?)\s+number\s+(?P<kind>total|frozen)$")
SAMPLE_HEADER_RE = re.compile(r"^(?P<sample>.+?)\s+\(n=(?P<n>[^)]+)\)$")
METADATA_ROW_LABELS = {
    "sample_id",
    "sample_name",
    "sample_long_name",
    "collection_start",
    "collection_end",
    "sample_type",
    "dilution",
    "air_volume_L",
    "filter_fraction_used",
    "suspension_volume_mL",
    "dry_mass_g",
    "sample_note",
}


def read_commented_preamble(path: str | Path) -> dict[str, str]:
    """Read leading '# key: value' preamble lines from a CSV file."""

    preamble: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                preamble[key.strip()] = value.strip()
    return preamble


def read_icescopy_temperature_sync_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read an Icescopy temperature-sync CSV with optional commented preamble."""

    preamble = read_commented_preamble(path)
    df = pd.read_csv(path, comment="#")
    return df, preamble


def read_icescopy_sample_metadata(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, SampleMetadata]]:
    """Read Icescopy commented session/sample metadata from a freeze-count CSV."""

    session_metadata: dict[str, str] = {}
    sample_rows: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                session_metadata[key.strip()] = value.strip()
                continue
            if "," in body:
                values = [value.strip() for value in body.split(",")]
                sample_rows[values[0]] = values[1:]

    metadata_by_sample_id: dict[str, SampleMetadata] = {}
    row_count = max((len(values) for values in sample_rows.values()), default=0)
    for index in range(row_count):
        raw_sample_metadata = {
            key: values[index] if index < len(values) else ""
            for key, values in sample_rows.items()
        }
        sample_id = (
            raw_sample_metadata.get("sample_name")
            or raw_sample_metadata.get("sample_id")
            or str(index)
        )
        metadata_by_sample_id[sample_id] = SampleMetadata(
            format_name=session_metadata.get("format_name", ""),
            file_version=session_metadata.get("file_version", ""),
            project_name=session_metadata.get("project_name", ""),
            user_name=session_metadata.get("user_name", ""),
            institution=session_metadata.get("institution", ""),
            date=session_metadata.get("analysis_date", session_metadata.get("date", "")),
            well_volume_uL=session_metadata.get("well_volume_uL"),
            reset_temperature_C=session_metadata.get("reset_temperature_C"),
            sample_id=sample_id,
            sample_name=raw_sample_metadata.get("sample_name", ""),
            sample_long_name=raw_sample_metadata.get("sample_long_name", ""),
            collection_start=raw_sample_metadata.get("collection_start", ""),
            collection_end=raw_sample_metadata.get("collection_end", ""),
            sample_type=raw_sample_metadata.get("sample_type", "other"),
            dilution=raw_sample_metadata.get("dilution"),
            air_volume_L=raw_sample_metadata.get("air_volume_L"),
            filter_fraction_used=raw_sample_metadata.get("filter_fraction_used"),
            suspension_volume_mL=raw_sample_metadata.get("suspension_volume_mL"),
            dry_mass_g=raw_sample_metadata.get("dry_mass_g"),
            raw_preamble=session_metadata,
            raw_sample_metadata=raw_sample_metadata,
        )
    return session_metadata, metadata_by_sample_id


def read_icescopy_freeze_count_timeseries_csv(path: str | Path) -> CountsTable:
    """Read an Icescopy freeze_count_timeseries CSV as a CountsTable with metadata."""

    df, _ = read_icescopy_temperature_sync_csv(path)
    counts = parse_icescopy_wide_temperature_sync(df)
    _, metadata = read_icescopy_sample_metadata(path)
    return CountsTable(
        sample_id=counts.sample_id,
        temperature_C=counts.temperature_C,
        n_total=counts.n_total,
        n_frozen=counts.n_frozen,
        time_s=counts.time_s,
        cycle=counts.cycle,
        observation_id=counts.observation_id,
        metadata=metadata,
    )


def sample_metadata_to_dataframe(
    metadata_by_sample_id: dict[str, SampleMetadata],
) -> pd.DataFrame:
    """Return sample metadata as a human-readable DataFrame."""

    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "sample_name": metadata.sample_name,
                "sample_long_name": metadata.sample_long_name,
                "sample_type": metadata.sample_type,
                "dilution": metadata.dilution,
                "well_volume_uL": metadata.well_volume_uL,
                "air_volume_L": metadata.air_volume_L,
                "filter_fraction_used": metadata.filter_fraction_used,
                "suspension_volume_mL": metadata.suspension_volume_mL,
                "dry_mass_g": metadata.dry_mass_g,
            }
            for sample_id, metadata in metadata_by_sample_id.items()
        ]
    )


def infer_icescopy_dilution_group_map(
    metadata_by_sample_id: dict[str, SampleMetadata],
) -> dict[str, str]:
    """Infer parent sample IDs from Icescopy long names like CRG_M1_13."""

    return {
        sample_id: _strip_trailing_numeric_token(
            metadata.sample_long_name or metadata.sample_name or sample_id
        )
        for sample_id, metadata in metadata_by_sample_id.items()
    }


def strip_temperature_sync_metadata_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sample metadata rows from temperature-sync data rows."""

    if df.empty:
        return df.copy(), df.iloc[0:0].copy()
    first_col = df.columns[0]
    labels = df[first_col].astype(str)
    metadata_mask = labels.isin(METADATA_ROW_LABELS)
    metadata_rows = df.loc[metadata_mask].copy()
    data_rows = df.loc[~metadata_mask].copy()
    return data_rows.reset_index(drop=True), metadata_rows.reset_index(drop=True)


def parse_icescopy_wide_temperature_sync(df: pd.DataFrame) -> CountsTable:
    """Convert an Icescopy wide temperature-sync table into a CountsTable."""

    data_rows, _ = strip_temperature_sync_metadata_rows(df)
    sample_columns: dict[str, dict[str, str]] = {}
    for column in data_rows.columns:
        match = SAMPLE_VALUE_RE.match(str(column))
        if not match:
            continue
        sample_key = match.group("sample").strip()
        kind = match.group("kind")
        sample_columns.setdefault(sample_key, {})[kind] = column

    records: list[dict[str, Any]] = []
    for sample_key, columns in sample_columns.items():
        if "total" not in columns or "frozen" not in columns:
            continue
        sample_id = _sample_id_from_header(sample_key)
        for row_index, row in data_rows.iterrows():
            total = _to_float_or_nan(row[columns["total"]])
            frozen = _to_float_or_nan(row[columns["frozen"]])
            if not np.isfinite(total) or not np.isfinite(frozen):
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "temperature_C": _to_float_or_nan(row.get("temperature_C", np.nan)),
                    "time_s": row_index,
                    "n_total": total,
                    "n_frozen": frozen,
                    "cycle": row.get("cycle", np.nan),
                    "observation_id": row.get("picture", row.get("image_name", row_index)),
                }
            )
    if not records:
        raise ValueError("No sample number total/frozen column pairs found")
    return CountsTable.from_dataframe(pd.DataFrame.from_records(records))


def _sample_id_from_header(sample_key: str) -> str:
    match = SAMPLE_HEADER_RE.match(sample_key)
    if match:
        return match.group("sample").strip()
    return sample_key.strip()


def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _strip_trailing_numeric_token(value: str) -> str:
    parts = str(value).rsplit("_", 1)
    if len(parts) != 2:
        return str(value)
    try:
        float(parts[1])
    except ValueError:
        return str(value)
    return parts[0]


def parse_olaf_frozen_at_temp_csv(
    df: pd.DataFrame,
    *,
    n_total_by_sample: dict[str, float],
) -> TemperatureFrozenFractionTable:
    """Convert an OLAF frozen_at_temp table into a TemperatureFrozenFractionTable."""

    if "degC" not in df:
        raise ValueError("OLAF frozen_at_temp data must include a 'degC' column")
    records: list[dict[str, Any]] = []
    for column in df.columns:
        if column == "degC":
            continue
        if column not in n_total_by_sample:
            raise KeyError(f"Missing n_total for sample column {column!r}")
        for _, row in df.iterrows():
            frozen = _to_float_or_nan(row[column])
            if not np.isfinite(frozen):
                continue
            records.append(
                {
                    "sample_id": column,
                    "temperature_C": float(row["degC"]),
                    "n_total": float(n_total_by_sample[column]),
                    "n_frozen": frozen,
                }
            )
    result = pd.DataFrame.from_records(records)
    return TemperatureFrozenFractionTable(
        sample_id=result["sample_id"].to_numpy(dtype=object),
        temperature_C=result["temperature_C"].to_numpy(dtype=float),
        n_total=result["n_total"].to_numpy(dtype=float),
        n_frozen=result["n_frozen"].to_numpy(dtype=float),
    )

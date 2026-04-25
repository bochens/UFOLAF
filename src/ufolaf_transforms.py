from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from ufolaf_math import (
    PROFILE_LIKELIHOOD_DROP_95,
    binomial_poisson_mle_with_profile_errors,
    bin_temperature,
    cumulative_inp_per_ml_with_errors_from_counts,
    differential_inp_per_ml_per_c_from_counts,
    normalize_inp_air,
    normalize_inp_soil,
    temperature_bin_edges,
    water_blank_corrected_counts,
)
from ufolaf_models import (
    CountsTable,
    CumulativeNucleusSpectrumTable,
    DifferentialNucleusSpectrumTable,
    MetadataLike,
    NormalizedInpSpectrumTable,
    SampleMetadata,
    SpectrumBasis,
    TemperatureDependentTable,
    TemperatureFrozenFractionTable,
)


CyclePolicy = Literal["single", "pooled", "preserve"]
TemperatureReductionMethod = Literal["max", "latest"]
StitchRowPolicy = Literal["olaf", "non_saturated", "partial_only", "all"]
OLAF_AGRESTI_COULL_UNCERTAIN_VALUES = 2


def apply_water_blank_correction(
    table: CountsTable | TemperatureFrozenFractionTable,
    water_blank_frozen: Any,
) -> CountsTable | TemperatureFrozenFractionTable:
    """Return a new count table after same-run water/DI blank correction.

    The water blank frozen count is subtracted from both n_frozen and n_total.
    The physical interpretation is that wells freezing in the water/DI blank are
    treated as unavailable sample wells at that temperature.
    """

    corrected_frozen, corrected_total = water_blank_corrected_counts(
        table.n_frozen,
        table.n_total,
        water_blank_frozen,
    )
    if isinstance(table, CountsTable):
        return CountsTable(
            sample_id=table.sample_id,
            temperature_C=table.temperature_C,
            n_total=corrected_total,
            n_frozen=corrected_frozen,
            time_s=table.time_s,
            cycle=table.cycle,
            observation_id=table.observation_id,
            metadata=table.metadata,
        )
    if isinstance(table, TemperatureFrozenFractionTable):
        return TemperatureFrozenFractionTable(
            sample_id=table.sample_id,
            temperature_C=table.temperature_C,
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            temperature_bin_left_C=table.temperature_bin_left_C,
            temperature_bin_right_C=table.temperature_bin_right_C,
            n_total=corrected_total,
            n_frozen=corrected_frozen,
            obs_count=table.obs_count,
            metadata=table.metadata,
        )
    raise TypeError("table must be a CountsTable or TemperatureFrozenFractionTable")


def counts_to_temperature_frozen_fraction(
    counts: CountsTable,
    *,
    step_C: float = 0.5,
    method: TemperatureReductionMethod = "max",
    cycle_policy: CyclePolicy = "single",
    cycle: Any | None = None,
) -> TemperatureFrozenFractionTable | list[TemperatureFrozenFractionTable]:
    """Reduce raw count observations to temperature-binned frozen-fraction table(s).

    ``method="max"`` selects by corrected frozen fraction within each bin, then
    carries forward the best fraction across colder bins while preserving paired
    n_total/n_frozen counts. ``method="latest"`` keeps the latest paired count
    row within each bin and does not force monotonicity.
    ``cycle_policy="single"`` returns one table and requires one cycle unless a
    specific ``cycle=...`` is selected. ``cycle_policy="pooled"`` reduces each
    cycle first, then sums n_frozen/n_total across cycles.
    ``cycle_policy="preserve"`` returns one normal-schema table per cycle.
    """

    if method not in ("max", "latest"):
        raise ValueError("method must be 'max' or 'latest'")
    if cycle_policy not in ("single", "pooled", "preserve"):
        raise ValueError("cycle_policy must be 'single', 'pooled', or 'preserve'")
    if cycle is not None and cycle_policy != "single":
        raise ValueError("cycle can only be selected when cycle_policy='single'")

    df = counts.to_dataframe()
    if df.empty:
        empty_table = _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=method,
            metadata=counts.metadata,
        )
        return [empty_table] if cycle_policy == "preserve" else empty_table

    df = _valid_counts_dataframe(df)
    if df.empty:
        empty_table = _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=method,
            metadata=counts.metadata,
        )
        return [empty_table] if cycle_policy == "preserve" else empty_table

    df = _with_cycle_key(df)
    cycle_keys = _cycle_keys(df)
    if cycle_policy == "single":
        selected_cycle = _selected_cycle_key(cycle_keys, cycle)
        selected_df = df[df["_ufolaf_cycle_key"] == selected_cycle].drop(
            columns="_ufolaf_cycle_key"
        )
        return _counts_dataframe_to_temperature_frozen_fraction(
            selected_df,
            step_C=step_C,
            method=method,
            temperature_bin_method=method,
            metadata=counts.metadata,
        )

    cycle_tables = [
        _counts_dataframe_to_temperature_frozen_fraction(
            cycle_df,
            step_C=step_C,
            method=method,
            temperature_bin_method=method,
            metadata=counts.metadata,
        )
        for _, cycle_df in _iter_cycle_dataframes(df)
    ]
    if cycle_policy == "preserve":
        return cycle_tables
    return _pool_temperature_frozen_fraction_tables(
        cycle_tables,
        step_C=step_C,
        temperature_bin_method=method,
        metadata=counts.metadata,
    )


def _counts_dataframe_to_temperature_frozen_fraction(
    df: pd.DataFrame,
    *,
    step_C: float,
    method: TemperatureReductionMethod,
    temperature_bin_method: str,
    metadata: MetadataLike,
) -> TemperatureFrozenFractionTable:
    if df.empty:
        return _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=temperature_bin_method,
            metadata=metadata,
        )

    df["temperature_C"] = bin_temperature(df["temperature_C"].to_numpy(copy=True), step_C)
    sort_columns = ["sample_id", "temperature_C"]
    ascending = [True, False]
    if "time_s" in df:
        sort_columns.append("time_s")
        ascending.append(True)
    df = df.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    grouped = df.groupby(["sample_id", "temperature_C"], sort=False)
    if method == "max":
        df["_ufolaf_fraction_frozen"] = df["n_frozen"] / df["n_total"]
        reduced = df.loc[grouped["_ufolaf_fraction_frozen"].idxmax()].copy()
        reduced = reduced.set_index(["sample_id", "temperature_C"])
        reduced = reduced[["n_total", "n_frozen"]]
        reduced["obs_count"] = grouped.size()
    else:
        reduced = grouped.tail(1).set_index(["sample_id", "temperature_C"])
        reduced = reduced[["n_total", "n_frozen"]]
        reduced["obs_count"] = grouped.size()
    reduced = reduced.reset_index()

    fixed_frames: list[pd.DataFrame] = []
    for _, sample_df in reduced.groupby("sample_id", sort=False):
        sample_df = sample_df.sort_values("temperature_C", ascending=False).copy()
        if method == "max":
            sample_df = _enforce_cumulative_fraction_pairs(sample_df)
        fixed_frames.append(sample_df)

    result = pd.concat(fixed_frames, ignore_index=True) if fixed_frames else reduced
    bin_left, bin_right = temperature_bin_edges(
        result["temperature_C"].to_numpy(dtype=float, copy=True),
        step_C,
    )
    return TemperatureFrozenFractionTable(
        sample_id=result["sample_id"].to_numpy(dtype=object, copy=True),
        temperature_C=result["temperature_C"].to_numpy(dtype=float, copy=True),
        temperature_bin_width_C=step_C,
        temperature_bin_method=temperature_bin_method,
        temperature_bin_left_C=bin_left,
        temperature_bin_right_C=bin_right,
        n_total=result["n_total"].to_numpy(dtype=float, copy=True),
        n_frozen=result["n_frozen"].to_numpy(dtype=float, copy=True),
        obs_count=result["obs_count"].to_numpy(dtype=int, copy=True),
        metadata=metadata,
    )


def _enforce_cumulative_fraction_pairs(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Carry forward the highest corrected fraction while preserving count pairs."""

    result = sample_df.copy()
    fraction = result["n_frozen"].to_numpy(dtype=float) / result["n_total"].to_numpy(dtype=float)
    best_positions: list[int] = []
    best_position = 0
    best_fraction = -np.inf
    for position, value in enumerate(fraction):
        if np.isfinite(value) and value >= best_fraction:
            best_fraction = value
            best_position = position
        best_positions.append(best_position)

    source_total = result["n_total"].to_numpy(dtype=float, copy=True)
    source_frozen = result["n_frozen"].to_numpy(dtype=float, copy=True)
    result["n_total"] = source_total[best_positions]
    result["n_frozen"] = source_frozen[best_positions]
    return result


def _valid_counts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        np.isfinite(df["temperature_C"])
        & np.isfinite(df["n_total"])
        & np.isfinite(df["n_frozen"])
        & (df["n_total"] > 0)
        & (df["n_frozen"] >= 0)
        & (df["n_frozen"] <= df["n_total"])
    ].copy()


def _with_cycle_key(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    if "cycle" not in keyed:
        keyed["_ufolaf_cycle_key"] = "missing"
        return keyed
    keyed["_ufolaf_cycle_key"] = [
        _normalize_cycle_key(value) for value in keyed["cycle"].to_numpy(dtype=object)
    ]
    return keyed


def _normalize_cycle_key(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        if pd.isna(value):
            return "missing"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)

    text = str(value).strip()
    if not text:
        return "missing"
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def _cycle_keys(df: pd.DataFrame) -> list[str]:
    return [str(value) for value in pd.unique(df["_ufolaf_cycle_key"])]


def _selected_cycle_key(cycle_keys: list[str], cycle: Any | None) -> str:
    if not cycle_keys:
        raise ValueError("No cycles found")
    if cycle is None:
        if len(cycle_keys) == 1:
            return cycle_keys[0]
        available = ", ".join(cycle_keys)
        raise ValueError(
            "Multiple cycles found. Pass cycle=... for cycle_policy='single', "
            "or use cycle_policy='pooled' or cycle_policy='preserve'. "
            f"Available cycles: {available}"
        )
    selected = _normalize_cycle_key(cycle)
    if selected not in cycle_keys:
        available = ", ".join(cycle_keys)
        raise ValueError(f"Requested cycle {selected!r} not found. Available cycles: {available}")
    return selected


def _iter_cycle_dataframes(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for cycle_key, cycle_df in df.groupby("_ufolaf_cycle_key", sort=False):
        frames.append((str(cycle_key), cycle_df.drop(columns="_ufolaf_cycle_key").copy()))
    return frames


def _pool_temperature_frozen_fraction_tables(
    tables: list[TemperatureFrozenFractionTable],
    *,
    step_C: float,
    temperature_bin_method: str,
    metadata: MetadataLike,
) -> TemperatureFrozenFractionTable:
    if not tables:
        return _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=temperature_bin_method,
            metadata=metadata,
        )

    combined = pd.concat([table.to_dataframe() for table in tables], ignore_index=True)
    aggregation: dict[str, tuple[str, str]] = {
        "n_total": ("n_total", "sum"),
        "n_frozen": ("n_frozen", "sum"),
    }
    if "obs_count" in combined:
        aggregation["obs_count"] = ("obs_count", "sum")
    for column in ("temperature_bin_left_C", "temperature_bin_right_C"):
        if column in combined:
            aggregation[column] = (column, "first")

    pooled = (
        combined.groupby(["sample_id", "temperature_C"], sort=False)
        .agg(**aggregation)
        .reset_index()
        .sort_values(["sample_id", "temperature_C"], ascending=[True, False])
    )
    return TemperatureFrozenFractionTable(
        sample_id=pooled["sample_id"].to_numpy(dtype=object, copy=True),
        temperature_C=pooled["temperature_C"].to_numpy(dtype=float, copy=True),
        temperature_bin_width_C=step_C,
        temperature_bin_method=temperature_bin_method,
        temperature_bin_left_C=_array_or_none(pooled, "temperature_bin_left_C"),
        temperature_bin_right_C=_array_or_none(pooled, "temperature_bin_right_C"),
        n_total=pooled["n_total"].to_numpy(dtype=float, copy=True),
        n_frozen=pooled["n_frozen"].to_numpy(dtype=float, copy=True),
        obs_count=pooled["obs_count"].to_numpy(dtype=int, copy=True)
        if "obs_count" in pooled
        else None,
        metadata=metadata,
    )


def _empty_temperature_frozen_fraction(
    *,
    step_C: float,
    temperature_bin_method: str,
    metadata: MetadataLike,
) -> TemperatureFrozenFractionTable:
    return TemperatureFrozenFractionTable(
        sample_id=np.array([], dtype=object),
        temperature_C=np.array([], dtype=float),
        temperature_bin_width_C=step_C,
        temperature_bin_method=temperature_bin_method,
        n_total=np.array([], dtype=float),
        n_frozen=np.array([], dtype=float),
        obs_count=np.array([], dtype=int),
        metadata=metadata,
    )


def temperature_frozen_fraction_to_differential_spectrum(
    table: TemperatureFrozenFractionTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
) -> DifferentialNucleusSpectrumTable:
    """Convert a temperature frozen-fraction table to differential k(T)."""

    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    if table.temperature_bin_width_C is None:
        raise ValueError("temperature_bin_width_C is required for k(T)")

    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in table.to_dataframe().groupby("sample_id", sort=False):
        sample_key = str(sample_id)
        if sample_key not in metadata_by_sample_id:
            raise KeyError(f"Missing SampleMetadata for sample_id {sample_key!r}")
        metadata = metadata_by_sample_id[sample_key]
        metadata.validate_for_count_to_suspension()
        sample_df = sample_df.sort_values("temperature_C", ascending=False).copy()
        k_value = differential_inp_per_ml_per_c_from_counts(
            sample_df["n_frozen"].to_numpy(dtype=float, copy=True),
            sample_df["n_total"].to_numpy(dtype=float, copy=True),
            metadata.well_volume_uL or 0.0,
            table.temperature_bin_width_C,
            metadata.dilution or 0.0,
        )
        out = pd.DataFrame(
            {
                "sample_id": sample_df["sample_id"].to_numpy(dtype=object, copy=True),
                "temperature_C": sample_df["temperature_C"].to_numpy(dtype=float, copy=True),
                "value": k_value,
                "value_unit": "INP_per_mL_suspension_per_C",
                "basis": "suspension",
                "qc_flag": np.where(np.isfinite(k_value), 0, 1),
            }
        )
        copy_temperature_metadata(sample_df, out)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return DifferentialNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension_per_C",
            basis="suspension",
            metadata=metadata_by_sample_id,
        )
    return DifferentialNucleusSpectrumTable(
        sample_id=result["sample_id"].to_numpy(dtype=object, copy=True),
        temperature_C=result["temperature_C"].to_numpy(dtype=float, copy=True),
        temperature_bin_width_C=_scalar_or_none(result, "temperature_bin_width_C"),
        temperature_bin_method=_scalar_string_or_empty(result, "temperature_bin_method"),
        temperature_bin_left_C=_array_or_none(result, "temperature_bin_left_C"),
        temperature_bin_right_C=_array_or_none(result, "temperature_bin_right_C"),
        value=result["value"].to_numpy(dtype=float, copy=True),
        value_unit="INP_per_mL_suspension_per_C",
        basis="suspension",
        qc_flag=result["qc_flag"].to_numpy(dtype=int, copy=True),
        metadata=metadata_by_sample_id,
    )


def temperature_frozen_fraction_to_cumulative_spectrum(
    table: TemperatureFrozenFractionTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    z: float = 1.96,
) -> CumulativeNucleusSpectrumTable:
    """Convert each dilution independently to cumulative K(T)."""

    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in table.to_dataframe().groupby("sample_id", sort=False):
        sample_key = str(sample_id)
        if sample_key not in metadata_by_sample_id:
            raise KeyError(f"Missing SampleMetadata for sample_id {sample_key!r}")
        metadata = metadata_by_sample_id[sample_key]
        metadata.validate_for_count_to_suspension()
        sample_df = sample_df.sort_values("temperature_C", ascending=False).copy()
        inp_per_ml, lower_error, upper_error, finite_mask = (
            cumulative_inp_per_ml_with_errors_from_counts(
                sample_df["n_frozen"].to_numpy(dtype=float, copy=True),
                sample_df["n_total"].to_numpy(dtype=float, copy=True),
                metadata.well_volume_uL or 0.0,
                metadata.dilution or 0.0,
                z=z,
            )
        )
        out = pd.DataFrame(
            {
                "sample_id": sample_df["sample_id"].to_numpy(dtype=object, copy=True),
                "temperature_C": sample_df["temperature_C"].to_numpy(dtype=float, copy=True),
                "value": inp_per_ml,
                "value_unit": "INP_per_mL_suspension",
                "basis": "suspension",
                "lower_ci": lower_error,
                "upper_ci": upper_error,
                "dilution_fold": metadata.dilution,
                "qc_flag": np.where(finite_mask, 0, 1),
            }
        )
        copy_temperature_metadata(sample_df, out)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=metadata_by_sample_id,
        )
    return CumulativeNucleusSpectrumTable.from_dataframe(
        result,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=metadata_by_sample_id,
    )


def temperature_frozen_fraction_to_stitched_cumulative_spectrum(
    table: TemperatureFrozenFractionTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str] = "sample_long_name",
    row_policy: StitchRowPolicy = "olaf",
    enforce_monotone: bool = False,
    z: float = 1.96,
) -> CumulativeNucleusSpectrumTable:
    """Stitch serial dilutions into one cumulative K(T) spectrum.

    The default follows OLAF's dilution-transition logic: start from the least
    diluted spectrum, inspect the last four valid overlap points before switching
    dilution, use the same confidence-interval decision tree, then use the next
    dilution for colder temperatures. ``row_policy="olaf"`` also applies OLAF's
    near-saturation pruning with a two-well Agresti-Coull margin.
    """

    if row_policy not in ("olaf", "non_saturated", "partial_only", "all"):
        raise ValueError("row_policy must be 'olaf', 'non_saturated', 'partial_only', or 'all'")
    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    per_dilution = temperature_frozen_fraction_to_cumulative_spectrum(
        table,
        metadata_by_sample_id,
        z=z,
    )
    source_df = per_dilution.to_dataframe()
    counts_df = table.to_dataframe()[
        ["sample_id", "temperature_C", "n_total", "n_frozen"]
    ].copy()
    if source_df.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata={},
        )

    source_df["source_sample_id"] = source_df["sample_id"].astype(str)
    counts_df["source_sample_id"] = counts_df["sample_id"].astype(str)
    source_df = source_df.merge(
        counts_df.drop(columns="sample_id"),
        on=["source_sample_id", "temperature_C"],
        how="left",
        validate="one_to_one",
    )
    source_df["stitch_group_id"] = [
        _sample_group_id(sample_id, metadata_by_sample_id[sample_id], sample_group_by)
        for sample_id in source_df["source_sample_id"]
    ]

    frames: list[pd.DataFrame] = []
    output_metadata: dict[str, SampleMetadata] = {}
    for group_id, group_df in source_df.groupby("stitch_group_id", sort=False):
        group_sample_ids = group_df["source_sample_id"].astype(str).unique()
        group_metadata = [metadata_by_sample_id[sample_id] for sample_id in group_sample_ids]
        output_metadata.setdefault(
            str(group_id),
            _combined_source_metadata(str(group_id), group_sample_ids, group_metadata, "stitch"),
        )
        stitched_group = _stitch_cumulative_group(
            group_df,
            str(group_id),
            row_policy=row_policy,
            enforce_monotone=enforce_monotone,
        )
        if not stitched_group.empty:
            frames.append(stitched_group)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=output_metadata,
        )
    result = result.sort_values(["sample_id", "temperature_C"], ascending=[True, False])
    return CumulativeNucleusSpectrumTable.from_dataframe(
        result,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=output_metadata,
    )


def temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum(
    table: TemperatureFrozenFractionTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str] = "sample_long_name",
    row_policy: Literal["all", "non_saturated", "partial_only"] = "all",
    confidence_drop: float = PROFILE_LIKELIHOOD_DROP_95,
) -> CumulativeNucleusSpectrumTable:
    """Combine dilution rows with a binomial-Poisson MLE at each temperature.

    Each row contributes observed frozen counts ``x_j`` and total wells ``n_j``.
    The metadata dilution fold maps a shared original-sample concentration K(T)
    to the per-well freezing probability for that dilution:

    p_j(K) = 1 - exp(-K * well_volume_mL / dilution_j)

    ``sample_group_by`` defines which sample rows are treated as dilutions of the
    same original sample. Use a dict for explicit sample_id -> group_id mapping,
    or one of the metadata fields listed in the type annotation.

    ``row_policy`` controls boundary rows: ``all`` keeps 0/n and n/n rows,
    ``non_saturated`` drops n/n rows, and ``partial_only`` keeps only 0 < x < n.
    """

    if row_policy not in ("all", "non_saturated", "partial_only"):
        raise ValueError("row_policy must be 'all', 'non_saturated', or 'partial_only'")
    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    source_df = table.to_dataframe()
    if source_df.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata={},
        )

    source_df["source_sample_id"] = source_df["sample_id"].astype(str)
    source_df["mle_group_id"] = [
        _sample_group_id(sample_id, metadata_by_sample_id[sample_id], sample_group_by)
        for sample_id in source_df["source_sample_id"]
    ]

    frames: list[pd.DataFrame] = []
    output_metadata: dict[str, SampleMetadata] = {}
    for (group_id, temperature_C), group_df in source_df.groupby(
        ["mle_group_id", "temperature_C"],
        sort=False,
    ):
        metadata_sample_ids = group_df["source_sample_id"].astype(str).to_numpy(copy=True)
        metadata_rows = [metadata_by_sample_id[sample_id] for sample_id in metadata_sample_ids]
        output_metadata.setdefault(
            str(group_id),
            _combined_source_metadata(str(group_id), metadata_sample_ids, metadata_rows, "mle"),
        )
        fit_df = _filter_mle_rows(group_df, row_policy)
        if fit_df.empty:
            value = np.nan
            lower_error = np.nan
            upper_error = np.nan
            finite = False
            dilution_fold = np.nan
        else:
            fit_sample_ids = fit_df["source_sample_id"].astype(str).to_numpy(copy=True)
            fit_metadata = [metadata_by_sample_id[sample_id] for sample_id in fit_sample_ids]
            well_volume_uL = _shared_well_volume_uL(fit_metadata, str(group_id))
            dilution = np.array(
                [metadata.dilution or 0.0 for metadata in fit_metadata],
                dtype=float,
            )
            if np.any(dilution <= 0):
                raise ValueError(f"All dilutions must be positive for MLE group {group_id!r}")

            value, lower_error, upper_error, finite = (
                binomial_poisson_mle_with_profile_errors(
                    fit_df["n_frozen"].to_numpy(dtype=float, copy=True),
                    fit_df["n_total"].to_numpy(dtype=float, copy=True),
                    well_volume_uL,
                    dilution,
                    confidence_drop=confidence_drop,
                )
            )
            dilution_fold = dilution[0] if len(pd.unique(dilution)) == 1 else np.nan
        out = pd.DataFrame(
            {
                "sample_id": [str(group_id)],
                "temperature_C": [float(temperature_C)],
                "value": [value],
                "value_unit": ["INP_per_mL_suspension"],
                "basis": ["suspension"],
                "lower_ci": [lower_error],
                "upper_ci": [upper_error],
                "dilution_fold": [dilution_fold],
                "qc_flag": [0 if finite else 1],
            }
        )
        copy_temperature_metadata(group_df.iloc[[0]], out)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=output_metadata,
        )
    result = result.sort_values(["sample_id", "temperature_C"], ascending=[True, False])
    return CumulativeNucleusSpectrumTable.from_dataframe(
        result,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=output_metadata,
    )


def cumulative_spectrum_to_normalized_inp_spectrum(
    spectrum: CumulativeNucleusSpectrumTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
) -> NormalizedInpSpectrumTable:
    """Normalize cumulative INP/mL suspension values to each sample basis."""

    metadata_by_sample_id = resolve_metadata_by_sample_id(spectrum, metadata_by_sample_id)
    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in spectrum.to_dataframe().groupby("sample_id", sort=False):
        sample_key = str(sample_id)
        if sample_key not in metadata_by_sample_id:
            raise KeyError(f"Missing SampleMetadata for sample_id {sample_key!r}")
        metadata = metadata_by_sample_id[sample_key]
        value, value_unit, basis = normalize_inp_by_metadata(
            sample_df["value"].to_numpy(dtype=float, copy=True),
            metadata,
        )
        lower_ci = None
        upper_ci = None
        if "lower_ci" in sample_df:
            lower_ci, _, _ = normalize_inp_by_metadata(
                sample_df["lower_ci"].to_numpy(dtype=float, copy=True),
                metadata,
            )
        if "upper_ci" in sample_df:
            upper_ci, _, _ = normalize_inp_by_metadata(
                sample_df["upper_ci"].to_numpy(dtype=float, copy=True),
                metadata,
            )
        out = pd.DataFrame(
            {
                "sample_id": sample_df["sample_id"].to_numpy(dtype=object, copy=True),
                "temperature_C": sample_df["temperature_C"].to_numpy(dtype=float, copy=True),
                "value": value,
                "value_unit": value_unit,
                "basis": basis,
                "inp_per_mL": sample_df["value"].to_numpy(dtype=float, copy=True),
                "lower_ci": lower_ci,
                "upper_ci": upper_ci,
                "dilution_fold": metadata.dilution,
                "qc_flag": sample_df["qc_flag"].to_numpy(dtype=int, copy=True)
                if "qc_flag" in sample_df
                else np.zeros(len(sample_df), dtype=int),
            }
        )
        copy_temperature_metadata(sample_df, out)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return NormalizedInpSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=spectrum.temperature_bin_width_C,
            temperature_bin_method=spectrum.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="unknown",
            basis="other",
            metadata=metadata_by_sample_id,
        )
    return NormalizedInpSpectrumTable.from_dataframe(result, metadata=metadata_by_sample_id)


def temperature_frozen_fraction_to_normalized_inp_spectrum(
    table: TemperatureFrozenFractionTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    z: float = 1.96,
) -> NormalizedInpSpectrumTable:
    """Convert temperature frozen fractions directly to normalized cumulative INP."""

    cumulative = temperature_frozen_fraction_to_cumulative_spectrum(
        table,
        metadata_by_sample_id,
        z=z,
    )
    return cumulative_spectrum_to_normalized_inp_spectrum(cumulative, metadata_by_sample_id)


def resolve_metadata_by_sample_id(
    table: TemperatureDependentTable | CountsTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None,
) -> dict[str, SampleMetadata]:
    """Return a copied sample metadata mapping for a table."""

    if metadata_by_sample_id is not None:
        return _normalize_metadata_mapping(metadata_by_sample_id)
    if isinstance(table.metadata, SampleMetadata):
        metadata = copy.deepcopy(table.metadata)
        if not metadata.sample_id:
            sample_ids = pd.Series(table.sample_id).astype(str).unique()
            if len(sample_ids) != 1:
                raise KeyError("SampleMetadata.sample_id is required for multi-sample tables")
            return {str(sample_ids[0]): metadata}
        return {metadata.sample_id: metadata}
    if isinstance(table.metadata, dict):
        return _normalize_metadata_mapping(table.metadata)
    raise KeyError("SampleMetadata is required on the table or as metadata_by_sample_id")


def copy_temperature_metadata(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """Copy temperature-bin metadata columns between intermediate frames."""

    for column in (
        "temperature_bin_width_C",
        "temperature_bin_method",
        "temperature_bin_left_C",
        "temperature_bin_right_C",
    ):
        if column in source:
            target[column] = source[column].to_numpy(copy=True)


def normalize_inp_by_metadata(
    inp_per_mL: np.ndarray,
    metadata: SampleMetadata,
) -> tuple[np.ndarray, str, SpectrumBasis]:
    """Normalize INP/mL suspension according to the sample type."""

    if metadata.sample_type == "air":
        metadata.validate_for_spectrum_normalization()
        return (
            normalize_inp_air(
                inp_per_mL,
                metadata.suspension_volume_mL or 0.0,
                metadata.air_volume_L or 0.0,
                metadata.filter_fraction_used or 0.0,
            ),
            "INP_per_L_air",
            "sampled_air",
        )
    if metadata.sample_type == "soil":
        metadata.validate_for_spectrum_normalization()
        return (
            normalize_inp_soil(
                inp_per_mL,
                metadata.suspension_volume_mL or 0.0,
                metadata.dry_mass_g or 0.0,
            ),
            "INP_per_g_dry_soil",
            "dry_soil",
        )
    return np.array(inp_per_mL, dtype=float, copy=True), "INP_per_mL_suspension", "suspension"


def _normalize_metadata_mapping(metadata: MetadataLike) -> dict[str, SampleMetadata]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict of SampleMetadata objects")
    copied = copy.deepcopy(metadata)
    for key, value in copied.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be sample_id strings")
        if not isinstance(value, SampleMetadata):
            raise TypeError("metadata values must be SampleMetadata objects")
    return copied


def _filter_mle_rows(
    df: pd.DataFrame,
    row_policy: Literal["all", "non_saturated", "partial_only"],
) -> pd.DataFrame:
    if row_policy == "all":
        return df
    if row_policy == "non_saturated":
        return df[df["n_frozen"] < df["n_total"]]
    return df[(df["n_frozen"] > 0) & (df["n_frozen"] < df["n_total"])]


def _stitch_cumulative_group(
    group_df: pd.DataFrame,
    group_id: str,
    *,
    row_policy: StitchRowPolicy,
    enforce_monotone: bool,
) -> pd.DataFrame:
    group_df = _prepare_olaf_stitch_frame(group_df, row_policy)
    if group_df.empty:
        return pd.DataFrame()

    temperatures = np.array(sorted(group_df["temperature_C"].unique(), reverse=True), dtype=float)
    dilutions = np.array(sorted(group_df["dilution_fold"].dropna().unique()), dtype=float)
    if len(dilutions) == 0:
        return pd.DataFrame.from_records(
            [
                _empty_stitch_row(group_id, float(temperature), group_df)
                for temperature in temperatures
            ]
        )
    if group_df.duplicated(["temperature_C", "dilution_fold"]).any():
        raise ValueError(f"Duplicate temperature/dilution rows found for stitch group {group_id!r}")

    value_matrix = _stitch_matrix(group_df, "value", temperatures, dilutions)
    lower_matrix = _stitch_matrix(group_df, "lower_ci", temperatures, dilutions)
    upper_matrix = _stitch_matrix(group_df, "upper_ci", temperatures, dilutions)

    first_dilution = dilutions[0]
    result = pd.DataFrame(
        {
            "temperature_C": temperatures,
            "dilution_fold": first_dilution,
            "value": value_matrix[first_dilution].to_numpy(dtype=float, copy=True),
            "lower_ci": lower_matrix[first_dilution].to_numpy(dtype=float, copy=True),
            "upper_ci": upper_matrix[first_dilution].to_numpy(dtype=float, copy=True),
            "qc_flag": 0,
        }
    )

    for next_dilution in dilutions[1:]:
        last_valid_indices = result.index[result["value"].notna()].to_series().tail(4).to_numpy()
        replacement_start = int(last_valid_indices[-1]) + 1 if len(last_valid_indices) else 0
        going_down = False
        for index in last_valid_indices:
            previous_value = _previous_finite_value(result["value"], int(index))
            current_value = result.at[int(index), "value"]
            current_is_going_down = (
                previous_value is not None
                and np.isfinite(current_value)
                and current_value < previous_value
            )
            _apply_olaf_overlap_decision(
                result,
                int(index),
                next_dilution,
                value_matrix[next_dilution],
                lower_matrix[next_dilution],
                upper_matrix[next_dilution],
                going_down=going_down,
            )
            going_down = going_down or current_is_going_down

        _replace_with_next_dilution(
            result,
            replacement_start,
            next_dilution,
            value_matrix[next_dilution],
            lower_matrix[next_dilution],
            upper_matrix[next_dilution],
        )

    if enforce_monotone:
        _enforce_monotone_stitch_result(result)

    rows = [
        _stitch_row_from_result(group_id, row, group_df)
        for _, row in result.iterrows()
    ]
    return pd.DataFrame.from_records(rows)


def _prepare_olaf_stitch_frame(
    group_df: pd.DataFrame,
    row_policy: StitchRowPolicy,
) -> pd.DataFrame:
    df = group_df.copy()
    df = df[np.isfinite(df["dilution_fold"])].copy()
    for column in ("value", "lower_ci", "upper_ci", "n_frozen", "n_total"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    valid = np.isfinite(df["value"])
    if row_policy == "olaf":
        valid &= df["n_frozen"] < (df["n_total"] - OLAF_AGRESTI_COULL_UNCERTAIN_VALUES)
    elif row_policy == "non_saturated":
        valid &= df["n_frozen"] < df["n_total"]
    elif row_policy == "partial_only":
        valid &= (df["n_frozen"] > 0) & (df["n_frozen"] < df["n_total"])
    elif row_policy != "all":
        raise ValueError("row_policy must be 'olaf', 'non_saturated', 'partial_only', or 'all'")

    df.loc[~valid, ["value", "lower_ci", "upper_ci"]] = np.nan
    return df.sort_values(["temperature_C", "dilution_fold"], ascending=[False, True])


def _stitch_matrix(
    df: pd.DataFrame,
    column: str,
    temperatures: np.ndarray,
    dilutions: np.ndarray,
) -> pd.DataFrame:
    return (
        df.pivot(index="temperature_C", columns="dilution_fold", values=column)
        .reindex(index=temperatures, columns=dilutions)
        .astype(float)
    )


def _apply_olaf_overlap_decision(
    result: pd.DataFrame,
    index: int,
    next_dilution: float,
    next_value: pd.Series,
    next_lower_ci: pd.Series,
    next_upper_ci: pd.Series,
    *,
    going_down: bool,
) -> None:
    previous_value = _previous_finite_value(result["value"], index)
    current_value = result.at[index, "value"]
    if previous_value is None or not np.isfinite(current_value):
        return
    if not (current_value < previous_value or going_down):
        return

    previous_lower_limit = previous_value - _finite_or_zero(result.at[index, "lower_ci"])
    next_candidate = float(next_value.iloc[index])
    current_below_limit = current_value < previous_lower_limit
    next_below_limit = np.isfinite(next_candidate) and next_candidate < previous_lower_limit
    if current_below_limit and next_below_limit:
        result.loc[index, ["dilution_fold", "value", "lower_ci", "upper_ci"]] = np.nan
        result.at[index, "qc_flag"] = int(result.at[index, "qc_flag"]) | 1
        return
    if (
        current_value > previous_value
        and np.isfinite(next_candidate)
        and next_candidate > previous_value
    ):
        _select_overlap_by_olaf_error_logic(
            result,
            index,
            next_dilution,
            next_value,
            next_lower_ci,
            next_upper_ci,
        )
        return
    if current_value >= previous_lower_limit:
        return
    if np.isfinite(next_candidate) and next_candidate >= previous_lower_limit:
        _set_result_row_from_next(
            result,
            index,
            next_dilution,
            next_value,
            next_lower_ci,
            next_upper_ci,
        )


def _select_overlap_by_olaf_error_logic(
    result: pd.DataFrame,
    index: int,
    next_dilution: float,
    next_value: pd.Series,
    next_lower_ci: pd.Series,
    next_upper_ci: pd.Series,
) -> None:
    previous_value = _previous_finite_value(result["value"], index)
    if previous_value is None:
        return
    previous_upper_error = _previous_finite_value(result["upper_ci"], index)
    if previous_upper_error is None:
        previous_upper_error = 0.0

    current_value = result.at[index, "value"]
    next_candidate = float(next_value.iloc[index])
    current_within_previous = previous_value + previous_upper_error > current_value
    next_within_previous = (
        np.isfinite(next_candidate)
        and previous_value + previous_upper_error > next_candidate
    )
    if current_within_previous and next_within_previous:
        current_upper_error = _finite_or_inf(result.at[index, "upper_ci"])
        next_upper_error = _finite_or_inf(next_upper_ci.iloc[index])
        if current_upper_error >= next_upper_error:
            _set_result_row_from_next(
                result,
                index,
                next_dilution,
                next_value,
                next_lower_ci,
                next_upper_ci,
            )
    elif current_within_previous:
        return
    elif next_within_previous:
        _set_result_row_from_next(
            result,
            index,
            next_dilution,
            next_value,
            next_lower_ci,
            next_upper_ci,
        )
    elif np.isfinite(next_candidate):
        result.at[index, "dilution_fold"] = next_dilution
        result.at[index, "value"] = (current_value + next_candidate) / 2.0
        result.at[index, "lower_ci"] = _rms_pair(
            result.at[index, "lower_ci"],
            next_lower_ci.iloc[index],
        )
        result.at[index, "upper_ci"] = _rms_pair(
            result.at[index, "upper_ci"],
            next_upper_ci.iloc[index],
        )


def _replace_with_next_dilution(
    result: pd.DataFrame,
    start_index: int,
    next_dilution: float,
    next_value: pd.Series,
    next_lower_ci: pd.Series,
    next_upper_ci: pd.Series,
) -> None:
    if start_index >= len(result):
        return
    target = result.index[start_index:]
    result.loc[target, "dilution_fold"] = next_dilution
    result.loc[target, "value"] = next_value.iloc[start_index:].to_numpy(dtype=float, copy=True)
    result.loc[target, "lower_ci"] = next_lower_ci.iloc[start_index:].to_numpy(
        dtype=float,
        copy=True,
    )
    result.loc[target, "upper_ci"] = next_upper_ci.iloc[start_index:].to_numpy(
        dtype=float,
        copy=True,
    )
    result.loc[target, "qc_flag"] = np.where(np.isfinite(result.loc[target, "value"]), 0, 1)


def _set_result_row_from_next(
    result: pd.DataFrame,
    index: int,
    next_dilution: float,
    next_value: pd.Series,
    next_lower_ci: pd.Series,
    next_upper_ci: pd.Series,
) -> None:
    result.at[index, "dilution_fold"] = next_dilution
    result.at[index, "value"] = float(next_value.iloc[index])
    result.at[index, "lower_ci"] = float(next_lower_ci.iloc[index])
    result.at[index, "upper_ci"] = float(next_upper_ci.iloc[index])
    result.at[index, "qc_flag"] = 0 if np.isfinite(result.at[index, "value"]) else 1


def _previous_finite_value(series: pd.Series, index: int) -> float | None:
    previous = series.iloc[:index].dropna()
    if previous.empty:
        return None
    value = float(previous.iloc[-1])
    return value if np.isfinite(value) else None


def _finite_or_zero(value: Any) -> float:
    converted = float(value)
    return converted if np.isfinite(converted) else 0.0


def _finite_or_inf(value: Any) -> float:
    converted = float(value)
    return converted if np.isfinite(converted) else np.inf


def _enforce_monotone_stitch_result(result: pd.DataFrame) -> None:
    previous_value = np.nan
    previous_upper_ci = np.nan
    for index in result.index:
        value = result.at[index, "value"]
        if not np.isfinite(value):
            continue
        if np.isfinite(previous_value) and value < previous_value:
            current_upper_ci = result.at[index, "upper_ci"]
            result.at[index, "value"] = previous_value
            result.at[index, "upper_ci"] = _rms_pair(current_upper_ci, previous_upper_ci)
            result.at[index, "qc_flag"] = int(result.at[index, "qc_flag"]) | 2
        previous_value = result.at[index, "value"]
        previous_upper_ci = result.at[index, "upper_ci"]


def _stitch_row_from_result(
    group_id: str,
    row: pd.Series,
    group_df: pd.DataFrame,
) -> dict[str, Any]:
    temperature_C = float(row["temperature_C"])
    selected_dilution = row["dilution_fold"]
    source_row = _source_row_for_stitch_result(group_df, temperature_C, selected_dilution)
    out = {
        "sample_id": group_id,
        "temperature_C": temperature_C,
        "value": float(row["value"]) if pd.notna(row["value"]) else np.nan,
        "value_unit": str(source_row.get("value_unit", "INP_per_mL_suspension")),
        "basis": str(source_row.get("basis", "suspension")),
        "lower_ci": float(row["lower_ci"]) if pd.notna(row["lower_ci"]) else np.nan,
        "upper_ci": float(row["upper_ci"]) if pd.notna(row["upper_ci"]) else np.nan,
        "dilution_fold": float(selected_dilution) if pd.notna(selected_dilution) else np.nan,
        "qc_flag": int(row["qc_flag"]) if pd.notna(row["qc_flag"]) else 0,
    }
    _copy_temperature_row_metadata(source_row, out)
    return out


def _source_row_for_stitch_result(
    group_df: pd.DataFrame,
    temperature_C: float,
    dilution_fold: Any,
) -> pd.Series:
    candidates = group_df[group_df["temperature_C"] == temperature_C]
    if pd.notna(dilution_fold):
        dilution_candidates = candidates[candidates["dilution_fold"] == float(dilution_fold)]
        if not dilution_candidates.empty:
            return dilution_candidates.iloc[0]
    if not candidates.empty:
        return candidates.iloc[0]
    return group_df.iloc[0]


def _stitch_row_from_series(group_id: str, row: pd.Series) -> dict[str, Any]:
    out = {
        "sample_id": group_id,
        "temperature_C": float(row["temperature_C"]),
        "value": float(row["value"]),
        "value_unit": str(row.get("value_unit", "INP_per_mL_suspension")),
        "basis": str(row.get("basis", "suspension")),
        "lower_ci": float(row["lower_ci"])
        if "lower_ci" in row and pd.notna(row["lower_ci"])
        else np.nan,
        "upper_ci": float(row["upper_ci"])
        if "upper_ci" in row and pd.notna(row["upper_ci"])
        else np.nan,
        "dilution_fold": float(row["dilution_fold"])
        if "dilution_fold" in row and pd.notna(row["dilution_fold"])
        else np.nan,
        "qc_flag": int(row["qc_flag"]) if "qc_flag" in row and pd.notna(row["qc_flag"]) else 0,
    }
    _copy_temperature_row_metadata(row, out)
    return out


def _empty_stitch_row(
    group_id: str,
    temperature_C: float,
    temp_df: pd.DataFrame,
) -> dict[str, Any]:
    out = {
        "sample_id": group_id,
        "temperature_C": temperature_C,
        "value": np.nan,
        "value_unit": "INP_per_mL_suspension",
        "basis": "suspension",
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "dilution_fold": np.nan,
        "qc_flag": 1,
    }
    if not temp_df.empty:
        _copy_temperature_row_metadata(temp_df.iloc[0], out)
    return out


def _copy_temperature_row_metadata(row: pd.Series, out: dict[str, Any]) -> None:
    for column in (
        "temperature_bin_width_C",
        "temperature_bin_method",
        "temperature_bin_left_C",
        "temperature_bin_right_C",
    ):
        if column in row and pd.notna(row[column]):
            out[column] = row[column]


def _rms_pair(left: Any, right: Any) -> float:
    values = np.array([left, right], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def _sample_group_id(
    sample_id: str,
    metadata: SampleMetadata,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"] | dict[str, str],
) -> str:
    if isinstance(sample_group_by, dict):
        if sample_id not in sample_group_by:
            raise KeyError(f"Missing sample group mapping for sample_id {sample_id!r}")
        group_id = str(sample_group_by[sample_id]).strip()
    else:
        if sample_group_by not in ("sample_id", "sample_name", "sample_long_name"):
            raise ValueError(
                "sample_group_by must be sample_id, sample_name, sample_long_name, or a dict"
            )
        group_id = str(getattr(metadata, sample_group_by)).strip()
    if not group_id:
        raise ValueError(f"Empty sample group id for sample_id {sample_id!r}")
    return group_id


def _shared_well_volume_uL(metadata_rows: list[SampleMetadata], group_id: str) -> float:
    values: list[float] = []
    for metadata in metadata_rows:
        metadata.validate_for_count_to_suspension()
        values.append(float(metadata.well_volume_uL or 0.0))
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"All wells in MLE group {group_id!r} must use the same well volume")
    return values[0]


def _combined_source_metadata(
    group_id: str,
    source_sample_ids: np.ndarray,
    source_metadata: list[SampleMetadata],
    source_prefix: str,
) -> SampleMetadata:
    base = copy.deepcopy(source_metadata[0])
    source_dilutions = [metadata.dilution for metadata in source_metadata]
    raw_sample_metadata = dict(base.raw_sample_metadata)
    raw_sample_metadata[f"{source_prefix}_source_sample_ids"] = ",".join(
        map(str, source_sample_ids)
    )
    raw_sample_metadata[f"{source_prefix}_source_dilutions"] = ",".join(
        "" if dilution is None else str(dilution) for dilution in source_dilutions
    )
    unique_dilutions = set(source_dilutions)
    return replace(
        base,
        sample_id=group_id,
        dilution=None if len(unique_dilutions) > 1 else source_dilutions[0],
        raw_sample_metadata=raw_sample_metadata,
    )


def _array_or_none(df: pd.DataFrame, column: str) -> np.ndarray | None:
    return df[column].to_numpy(dtype=float, copy=True) if column in df else None


def _scalar_or_none(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    values = pd.Series(df[column]).dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"{column} must have one value")
    return float(values[0])


def _scalar_string_or_empty(df: pd.DataFrame, column: str) -> str:
    if column not in df:
        return ""
    values = pd.Series(df[column]).dropna().unique()
    if len(values) == 0:
        return ""
    if len(values) > 1:
        raise ValueError(f"{column} must have one value")
    return str(values[0])

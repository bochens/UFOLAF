from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ufolaf_models import NormalizedInpSpectrumTable


def propagate_uncertainty_rss(
    sample_error: np.ndarray,
    blank_error: np.ndarray,
) -> np.ndarray:
    """Propagate independent sample and blank uncertainties by root-sum-of-squares."""

    sample = np.asarray(sample_error, dtype=float)
    blank = np.asarray(blank_error, dtype=float)
    sample, blank = np.broadcast_arrays(sample, blank)
    return np.sqrt(sample**2 + blank**2)


def average_blank_spectra(
    blank_spectra: list[NormalizedInpSpectrumTable],
    *,
    value_method: Literal["mean", "median"] = "mean",
) -> NormalizedInpSpectrumTable:
    """Average replicate blank spectra by exact temperature."""

    if not blank_spectra:
        raise ValueError("At least one blank spectrum is required")
    frames = []
    for idx, spectrum in enumerate(blank_spectra):
        frame = spectrum.to_dataframe().copy()
        frame["blank_replicate"] = idx
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["value_unit"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectra must use one value_unit")
    if "basis" in combined and combined["basis"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectra must use one basis")

    value_agg = "mean" if value_method == "mean" else "median"
    grouped = combined.groupby("temperature_C", as_index=False).agg(
        value=("value", value_agg),
        inp_per_mL=("inp_per_mL", value_agg) if "inp_per_mL" in combined else ("value", value_agg),
        lower_ci=("lower_ci", _rms_or_nan) if "lower_ci" in combined else ("value", _nan_like),
        upper_ci=("upper_ci", _rms_or_nan) if "upper_ci" in combined else ("value", _nan_like),
        replicate_count=("value", "size"),
        value_unit=("value_unit", "first"),
        basis=("basis", "first") if "basis" in combined else ("value_unit", _default_basis),
        temperature_bin_width_C=("temperature_bin_width_C", "first")
        if "temperature_bin_width_C" in combined
        else ("value", _nan_like),
        temperature_bin_method=("temperature_bin_method", "first")
        if "temperature_bin_method" in combined
        else ("value_unit", _empty_string),
        temperature_bin_left_C=("temperature_bin_left_C", "first")
        if "temperature_bin_left_C" in combined
        else ("value", _nan_like),
        temperature_bin_right_C=("temperature_bin_right_C", "first")
        if "temperature_bin_right_C" in combined
        else ("value", _nan_like),
    )
    return NormalizedInpSpectrumTable(
        sample_id=np.repeat("blank", len(grouped)),
        temperature_C=grouped["temperature_C"].to_numpy(dtype=float),
        temperature_bin_width_C=_single_positive_or_none(grouped["temperature_bin_width_C"]),
        temperature_bin_method=str(grouped["temperature_bin_method"].iloc[0])
        if str(grouped["temperature_bin_method"].iloc[0]) != "nan"
        else "",
        temperature_bin_left_C=grouped["temperature_bin_left_C"].to_numpy(dtype=float)
        if grouped["temperature_bin_left_C"].notna().any()
        else None,
        temperature_bin_right_C=grouped["temperature_bin_right_C"].to_numpy(dtype=float)
        if grouped["temperature_bin_right_C"].notna().any()
        else None,
        value=grouped["value"].to_numpy(dtype=float),
        value_unit=str(grouped["value_unit"].iloc[0]),
        basis=str(grouped["basis"].iloc[0]),
        inp_per_mL=grouped["inp_per_mL"].to_numpy(dtype=float),
        lower_ci=grouped["lower_ci"].to_numpy(dtype=float),
        upper_ci=grouped["upper_ci"].to_numpy(dtype=float),
        replicate_count=grouped["replicate_count"].to_numpy(dtype=int),
        is_extrapolated=np.repeat(False, len(grouped)),
    )


def _rms_or_nan(values: pd.Series) -> float:
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def _nan_like(_: pd.Series) -> float:
    return np.nan


def _default_basis(_: pd.Series) -> str:
    return "other"


def _empty_string(_: pd.Series) -> str:
    return ""


def _single_positive_or_none(values: pd.Series) -> float | None:
    clean = values.dropna().unique()
    if len(clean) == 0:
        return None
    if len(clean) > 1:
        raise ValueError("Blank spectra must use one temperature_bin_width_C")
    value = float(clean[0])
    return value if value > 0 else None


def align_spectra_on_temperature(
    sample: NormalizedInpSpectrumTable,
    blank: NormalizedInpSpectrumTable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample and blank rows aligned to exact shared temperatures."""

    sample_df = sample.to_dataframe().set_index("temperature_C", drop=False)
    blank_df = blank.to_dataframe().set_index("temperature_C", drop=False)
    common = sample_df.index.intersection(blank_df.index)
    if common.empty:
        raise ValueError("Sample and blank spectra have no common temperatures")
    common = common.sort_values(ascending=False)
    return sample_df.loc[common].copy(), blank_df.loc[common].copy()


def subtract_blank_spectrum(
    sample: NormalizedInpSpectrumTable,
    blank: NormalizedInpSpectrumTable,
    *,
    clamp_zero: bool = True,
) -> NormalizedInpSpectrumTable:
    """Subtract a blank spectrum from a sample spectrum on shared temperatures."""

    sample_df, blank_df = align_spectra_on_temperature(sample, blank)
    if sample_df["value_unit"].nunique(dropna=True) != 1:
        raise ValueError("Sample spectrum must use one value_unit for blank subtraction")
    if blank_df["value_unit"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectrum must use one value_unit for blank subtraction")
    if sample_df["value_unit"].iloc[0] != blank_df["value_unit"].iloc[0]:
        raise ValueError("Sample and blank spectra must use the same value_unit")
    if "basis" in sample_df and "basis" in blank_df:
        if sample_df["basis"].iloc[0] != blank_df["basis"].iloc[0]:
            raise ValueError("Sample and blank spectra must use the same basis")

    corrected_value = sample_df["value"].to_numpy(dtype=float) - blank_df["value"].to_numpy(
        dtype=float
    )
    if clamp_zero:
        corrected_value = np.maximum(corrected_value, 0.0)

    lower_ci = None
    upper_ci = None
    if "lower_ci" in sample_df and "lower_ci" in blank_df:
        lower_ci = propagate_uncertainty_rss(
            sample_df["lower_ci"].to_numpy(dtype=float),
            blank_df["lower_ci"].to_numpy(dtype=float),
        )
    if "upper_ci" in sample_df and "upper_ci" in blank_df:
        upper_ci = propagate_uncertainty_rss(
            sample_df["upper_ci"].to_numpy(dtype=float),
            blank_df["upper_ci"].to_numpy(dtype=float),
        )

    corrected_per_ml = None
    if "inp_per_mL" in sample_df and "inp_per_mL" in blank_df:
        corrected_per_ml = sample_df["inp_per_mL"].to_numpy(dtype=float) - blank_df[
            "inp_per_mL"
        ].to_numpy(dtype=float)
        if clamp_zero:
            corrected_per_ml = np.maximum(corrected_per_ml, 0.0)

    return NormalizedInpSpectrumTable(
        sample_id=sample_df["sample_id"].to_numpy(dtype=object),
        temperature_C=sample_df["temperature_C"].to_numpy(dtype=float),
        temperature_bin_width_C=float(sample_df["temperature_bin_width_C"].dropna().iloc[0])
        if "temperature_bin_width_C" in sample_df
        and not sample_df["temperature_bin_width_C"].dropna().empty
        else None,
        temperature_bin_method=str(sample_df["temperature_bin_method"].dropna().iloc[0])
        if "temperature_bin_method" in sample_df
        and not sample_df["temperature_bin_method"].dropna().empty
        else "",
        temperature_bin_left_C=sample_df["temperature_bin_left_C"].to_numpy(dtype=float)
        if "temperature_bin_left_C" in sample_df
        else None,
        temperature_bin_right_C=sample_df["temperature_bin_right_C"].to_numpy(dtype=float)
        if "temperature_bin_right_C" in sample_df
        else None,
        value=corrected_value,
        value_unit=str(sample_df["value_unit"].iloc[0]),
        basis=str(sample_df["basis"].iloc[0]) if "basis" in sample_df else "other",
        inp_per_mL=corrected_per_ml,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        dilution_fold=sample_df["dilution_fold"].to_numpy(dtype=float)
        if "dilution_fold" in sample_df
        else None,
        qc_flag=np.repeat(0, len(sample_df)),
        is_extrapolated=sample_df["is_extrapolated"].to_numpy(dtype=bool)
        if "is_extrapolated" in sample_df
        else None,
        correction_state="blank_corrected",
    )


def extrapolate_blank_tail(
    blank: NormalizedInpSpectrumTable,
    target_temperatures_C: np.ndarray,
    *,
    tail_points: int = 4,
) -> NormalizedInpSpectrumTable:
    """Linearly extrapolate a blank spectrum to colder target temperatures."""

    if tail_points < 2:
        raise ValueError("tail_points must be at least 2")
    blank_df = blank.to_dataframe().sort_values("temperature_C", ascending=False)
    target = np.asarray(target_temperatures_C, dtype=float)
    if target.size == 0:
        return blank
    min_blank_temp = blank_df["temperature_C"].min()
    extrap_temps = np.array([temp for temp in target if temp < min_blank_temp], dtype=float)
    if extrap_temps.size == 0:
        return blank
    tail = blank_df.tail(tail_points)
    if len(tail) < 2:
        raise ValueError("At least two blank points are required for extrapolation")
    fit = np.polyfit(tail["temperature_C"].to_numpy(dtype=float), tail["value"].to_numpy(float), 1)
    extrap_values = fit[0] * extrap_temps + fit[1]

    lower_ratio = _safe_error_ratio(tail, "lower_ci")
    upper_ratio = _safe_error_ratio(tail, "upper_ci")
    extrap_df = pd.DataFrame(
        {
            "sample_id": "blank",
            "temperature_C": extrap_temps,
            "value": extrap_values,
            "value_unit": blank_df["value_unit"].iloc[0],
            "basis": blank_df["basis"].iloc[0] if "basis" in blank_df else "other",
            "lower_ci": extrap_values * lower_ratio,
            "upper_ci": extrap_values * upper_ratio,
            "replicate_count": 0,
            "is_extrapolated": True,
        }
    )
    if "temperature_bin_width_C" in blank_df:
        width_values = blank_df["temperature_bin_width_C"].dropna().unique()
        if len(width_values) == 1:
            width = float(width_values[0])
            extrap_df["temperature_bin_width_C"] = width
            extrap_df["temperature_bin_left_C"] = extrap_temps - width / 2.0
            extrap_df["temperature_bin_right_C"] = extrap_temps + width / 2.0
    if "temperature_bin_method" in blank_df:
        method_values = blank_df["temperature_bin_method"].dropna().unique()
        if len(method_values) == 1:
            extrap_df["temperature_bin_method"] = str(method_values[0])
    if "inp_per_mL" in blank_df:
        fit_ml = np.polyfit(
            tail["temperature_C"].to_numpy(dtype=float),
            tail["inp_per_mL"].to_numpy(dtype=float),
            1,
        )
        extrap_df["inp_per_mL"] = fit_ml[0] * extrap_temps + fit_ml[1]
    combined = pd.concat([blank_df, extrap_df], ignore_index=True).sort_values(
        "temperature_C", ascending=False
    )
    return NormalizedInpSpectrumTable.from_dataframe(combined)


def _safe_error_ratio(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return np.nan
    values = df[column].to_numpy(dtype=float)
    base = np.maximum(df["value"].to_numpy(dtype=float), 1e-12)
    return float(np.nanmean(values / base))

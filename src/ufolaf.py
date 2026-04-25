"""Public UFOLAF API.

This module is intentionally a thin facade. Implementation functions keep their
long descriptive names in the underlying modules; this file exposes the shorter
names intended for normal analysis scripts and notebooks.
"""

from ufolaf_adapters import (
    infer_icescopy_dilution_group_map as infer_dilution_groups,
    parse_olaf_frozen_at_temp_csv,
    read_icescopy_freeze_count_timeseries_csv as read_icescopy_counts,
    read_icescopy_sample_metadata as read_icescopy_metadata,
    read_icescopy_temperature_sync_csv as read_icescopy_temperature_sync,
    sample_metadata_to_dataframe as sample_metadata_dataframe,
)
from ufolaf_models import (
    CountsTable,
    CumulativeNucleusSpectrumTable,
    DifferentialNucleusSpectrumTable,
    NormalizedInpSpectrumTable,
    SampleMetadata,
    TemperatureDependentTable,
    TemperatureFrozenFractionTable,
)
from ufolaf_transforms import (
    apply_water_blank_correction as apply_water_blank,
    counts_to_temperature_frozen_fraction as fraction_frozen,
    cumulative_spectrum_to_normalized_inp_spectrum as normalize_spec,
    temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum as cumulative_spec_mle,
    temperature_frozen_fraction_to_cumulative_spectrum as cumulative_spec,
    temperature_frozen_fraction_to_differential_spectrum as differential_spec,
    temperature_frozen_fraction_to_stitched_cumulative_spectrum as cumulative_spec_stitch,
)

__all__ = [
    "CountsTable",
    "CumulativeNucleusSpectrumTable",
    "DifferentialNucleusSpectrumTable",
    "NormalizedInpSpectrumTable",
    "SampleMetadata",
    "TemperatureDependentTable",
    "TemperatureFrozenFractionTable",
    "apply_water_blank",
    "cumulative_spec",
    "cumulative_spec_mle",
    "cumulative_spec_stitch",
    "differential_spec",
    "fraction_frozen",
    "infer_dilution_groups",
    "normalize_spec",
    "parse_olaf_frozen_at_temp_csv",
    "read_icescopy_counts",
    "read_icescopy_metadata",
    "read_icescopy_temperature_sync",
    "sample_metadata_dataframe",
]

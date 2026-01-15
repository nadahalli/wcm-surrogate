"""
Load and process experimental E. coli data for model validation.

Data sources:
- Taheri-Araghi et al. (2015) Curr Biol: Cell size homeostasis
- Si et al. (2017) Curr Biol: Nutrient-dependent cell size
- Scott et al. (2010) Science: Bacterial growth laws
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentalCondition:
    """A single experimental condition with measured values."""
    growth_rate_per_hr: float      # Growth rate in 1/hour
    growth_rate_per_min: float     # Growth rate in 1/minute
    doubling_time_min: float       # Doubling time in minutes
    avg_volume_um3: Optional[float] = None
    birth_volume_um3: Optional[float] = None
    rna_protein_ratio: Optional[float] = None
    source: str = ""
    condition: str = ""


def get_data_dir() -> Path:
    """Get the data directory path."""
    # Try relative to this file first
    module_dir = Path(__file__).parent.parent.parent.parent
    data_dir = module_dir / "data"
    if data_dir.exists():
        return data_dir
    # Fallback to current working directory
    return Path("data")


def load_taheri_araghi() -> list[ExperimentalCondition]:
    """
    Load Taheri-Araghi et al. (2015) data.

    This dataset contains growth rate and cell volume measurements
    across 7 different nutrient conditions.

    Returns:
        List of ExperimentalCondition objects
    """
    data_path = get_data_dir() / "taheri-araghi_2015_data.csv"
    df = pd.read_csv(data_path)

    conditions = []
    for _, row in df.iterrows():
        growth_rate_hr = row["growth_rate_per_hr"]
        growth_rate_min = growth_rate_hr / 60
        doubling_time = np.log(2) / growth_rate_min if growth_rate_min > 0 else np.inf

        conditions.append(ExperimentalCondition(
            growth_rate_per_hr=growth_rate_hr,
            growth_rate_per_min=growth_rate_min,
            doubling_time_min=doubling_time,
            avg_volume_um3=row["avg_cell_volume_um3"],
            birth_volume_um3=row["avg_birth_volume_um3"],
            source="Taheri-Araghi 2015",
            condition=f"nutrient_{int(row['nutrient_type'])}"
        ))

    return conditions


def load_si_2017() -> list[ExperimentalCondition]:
    """
    Load Si et al. (2017) data.

    This dataset contains more detailed measurements including
    cell dimensions, RNA/protein ratios, and growth rates.

    Returns:
        List of ExperimentalCondition objects
    """
    data_path = get_data_dir() / "si_2017_data.csv"
    df = pd.read_csv(data_path)

    conditions = []
    for _, row in df.iterrows():
        growth_rate_hr = row["growth_rate_per_hr"]
        growth_rate_min = growth_rate_hr / 60
        doubling_time = np.log(2) / growth_rate_min if growth_rate_min > 0 else np.inf

        conditions.append(ExperimentalCondition(
            growth_rate_per_hr=growth_rate_hr,
            growth_rate_per_min=growth_rate_min,
            doubling_time_min=doubling_time,
            avg_volume_um3=row.get("source_cell_volume_um3"),
            rna_protein_ratio=row.get("RNA_prot_ratio"),
            source="Si 2017",
            condition=f"nutrient_{int(row['nutrient_type'])}_cm_{row['cm_uM']}"
        ))

    return conditions


def load_scott_2010() -> list[ExperimentalCondition]:
    """
    Load Scott et al. (2010) data.

    Classic bacterial growth laws paper showing relationship
    between growth rate and ribosomal fraction.

    Returns:
        List of ExperimentalCondition objects
    """
    data_path = get_data_dir() / "scott_2010_data.csv"
    df = pd.read_csv(data_path)

    conditions = []
    for _, row in df.iterrows():
        growth_rate_hr = row["growth_rate_per_hr"]
        growth_rate_min = growth_rate_hr / 60
        doubling_time = np.log(2) / growth_rate_min if growth_rate_min > 0 else np.inf

        conditions.append(ExperimentalCondition(
            growth_rate_per_hr=growth_rate_hr,
            growth_rate_per_min=growth_rate_min,
            doubling_time_min=doubling_time,
            rna_protein_ratio=row.get("measured_R_P_ratio"),
            source="Scott 2010",
            condition=row.get("media_details", "")
        ))

    return conditions


def get_growth_rate_range() -> tuple[float, float]:
    """
    Get the physiological range of E. coli growth rates from experimental data.

    Returns:
        (min_rate, max_rate) in 1/minute
    """
    all_conditions = load_taheri_araghi() + load_si_2017()
    rates = [c.growth_rate_per_min for c in all_conditions if c.growth_rate_per_min > 0]
    return min(rates), max(rates)


def get_experimental_summary() -> dict:
    """
    Get summary statistics from all experimental datasets.

    Returns:
        Dictionary with summary statistics
    """
    taheri = load_taheri_araghi()
    si = load_si_2017()
    scott = load_scott_2010()

    taheri_rates = [c.growth_rate_per_hr for c in taheri]
    taheri_volumes = [c.avg_volume_um3 for c in taheri if c.avg_volume_um3]

    si_rates = [c.growth_rate_per_hr for c in si]
    si_volumes = [c.avg_volume_um3 for c in si if c.avg_volume_um3]

    return {
        "taheri_araghi_2015": {
            "n_conditions": len(taheri),
            "growth_rate_range_per_hr": (min(taheri_rates), max(taheri_rates)),
            "volume_range_um3": (min(taheri_volumes), max(taheri_volumes)),
        },
        "si_2017": {
            "n_conditions": len(si),
            "growth_rate_range_per_hr": (min(si_rates), max(si_rates)),
            "volume_range_um3": (min(si_volumes), max(si_volumes)),
        },
        "scott_2010": {
            "n_conditions": len(scott),
            "growth_rate_range_per_hr": (
                min(c.growth_rate_per_hr for c in scott),
                max(c.growth_rate_per_hr for c in scott)
            ),
        }
    }

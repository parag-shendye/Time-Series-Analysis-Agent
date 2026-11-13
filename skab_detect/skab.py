"""SKAB file loading. Ground-truth labels are returned separately from the sensor frame."""
from pathlib import Path

import pandas as pd

SENSORS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]
GROUPS = ("valve1", "valve2", "other", "anomaly-free")


def list_files(root: Path | str, groups=GROUPS) -> list[str]:
    """File ids such as 'valve1/3', in numeric order within each group."""
    root = Path(root)
    return [f"{g}/{p.stem}" for g in groups
            for p in sorted((root / g).glob("*.csv"), key=lambda p: (len(p.stem), p.stem))]


def load(root: Path | str, file_id: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (sensor frame, truth frame). Truth has anomaly/changepoint columns, or is None."""
    group, name = file_id.split("/")
    df = pd.read_csv(Path(root) / group / f"{name}.csv", sep=";",
                     index_col="datetime", parse_dates=True).sort_index()
    cols = [c for c in ("anomaly", "changepoint") if c in df]
    return df[SENSORS], (df[cols] if cols else None)


def split(files: list[str], test_every: int = 4) -> tuple[list[str], list[str]]:
    """Deterministic file-level split: every Nth file is held out."""
    test = [f for i, f in enumerate(files) if i % test_every == test_every - 1]
    return [f for f in files if f not in test], test

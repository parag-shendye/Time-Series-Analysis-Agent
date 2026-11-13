"""SKAB loader. Ground-truth labels are kept separate from the sensor frame."""
from dataclasses import dataclass

import pandas as pd
from skab_detect import skab
from skab_detect.skab import GROUPS, SENSORS, split  # noqa: F401  (re-exported for callers)

from .config import SKAB_DATA


@dataclass
class Series:
    file_id: str            # e.g. "valve1/3"
    group: str
    X: pd.DataFrame         # datetime index, sensor columns only
    truth: pd.DataFrame | None  # anomaly/changepoint columns; for scoring only


def list_files() -> list[str]:
    return skab.list_files(SKAB_DATA)


def load(file_id: str) -> Series:
    X, truth = skab.load(SKAB_DATA, file_id)
    return Series(file_id, file_id.split("/")[0], X, truth)

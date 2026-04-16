"""
data_loading.py
---------------
Shared data loading and preprocessing utilities for Raman spectroscopy
classification models.  Handles raw spectral data ingestion, cosmic-ray
removal, baseline correction, and grouping into per-subject arrays.

Expected input files (set paths in the constants below):
    SPECTRA_PATH  – CSV of raw interpolated Raman spectra
                    rows = individual spectra,
                    columns = wavenumber bins 900-1800 cm⁻¹ (901 columns)
    LABELS_PATH   – CSV with columns [Name, Key]
                    Name = subject identifier string
                    Key  = integer class label
                           1 = Healthy, 2 = Neuropathy, 3 = Myopathy
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from orpl.cosmic_ray import crfilter_multi
from orpl.baseline_removal import bubblefill

# ── Data paths (update before running) ────────────────────────────────────────
SPECTRA_PATH = "/path/to/raw_interpolated_spectra.csv"
LABELS_PATH  = "/path/to/subject_labels.csv"

# ── Spectral axis ─────────────────────────────────────────────────────────────
WAVENUMBER_AXIS = np.arange(900, 1801)   # 901 wavenumber bins (cm⁻¹)
N_WAVENUMBERS   = len(WAVENUMBER_AXIS)   # 901

# ── Class label definitions ───────────────────────────────────────────────────
CLASS_MAP   = {1: "Healthy", 2: "Neuropathy", 3: "Myopathy"}
CLASS_NAMES = ["Healthy", "Neuropathy", "Myopathy"]   # index 0,1,2 → Key 1,2,3


def load_raw_data():
    """Load raw spectral matrix and subject label table from CSV.

    Returns
    -------
    raw_spectra : np.ndarray, shape (N, 901)
    labels_df   : pd.DataFrame, columns [Name, Key]
    """
    labels_df   = pd.read_csv(LABELS_PATH,  names=["Name", "Key"])
    raw_spectra = pd.read_csv(SPECTRA_PATH, header=None).to_numpy()
    return raw_spectra, labels_df


def preprocess_spectra(raw_spectra, min_bubble_widths=200):
    """Cosmic-ray removal followed by BubbleFill baseline correction."""
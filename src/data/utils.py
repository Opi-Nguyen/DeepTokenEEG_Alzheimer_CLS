

#knowledge


CHANNELS_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"
]

CHANNELS_19_LEGACY = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

BANDS = ["fullband", "delta", "theta", "alpha", "beta", "gamma"]
BAND_PRESETS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 16.0),
    "beta":  (16.0, 32.0),
    "gamma": (32.0, 45.0),
    "fullband": (0.5, 45.0),
}
SWT_RECONSTRUCTION_MAP = {
    "delta": {"A4"},
    "theta": {"D4"},
    "alpha": {"D3"},
    "beta":  {"D2"},
    "gamma": {"D1"},
    "fullband": {"A0"},
}

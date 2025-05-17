import os
from pathlib import Path

import sys

path = Path(__file__).parent.parent

# Data Path
ETH3D_DATA_PATH = Path(path / "data/datasets/ETH3D")
MIP_NERF_360_DATA_PATH = Path(path / "data/datasets/MipNerf360")
TANKS_AND_TEMPLES_DATA_PATH = Path(path / "data/datasets/TanksAndTemples")
LAMAR_HGE_DATA_PATH = Path(path / "data/datasets/LaMAR/HGE/sessions/map/raw_data/")
LAMAR_CAB_DATA_PATH = Path(path / "data/datasets/LaMAR/CAB/sessions/map/raw_data/")
LAMAR_LIN_DATA_PATH = Path(path / "data/datasets/LaMAR/LIN/sessions/map/raw_data/")

ETH3D_SCENES = (
        "courtyard",
        "delivery_area",
        "electro",
        "facade",
        "kicker",
        "meadow",
        "office",
        "pipes",
        "playground",
        "relief",
        "relief_2",
        "terrace",
        "terrains",
)

MIP_NERF_360_SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump"
)

TANKS_AND_TEMPLES_SCENES = (
    "Barn",
    "Caterpillar",
    "Church",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck"
)

LAMAR_HGE_SCENES = sorted([p.name for p in LAMAR_HGE_DATA_PATH.glob("ios*") if p.is_dir()])
LAMAR_CAB_SCENES = sorted([p.name for p in LAMAR_CAB_DATA_PATH.glob("ios*") if p.is_dir()])
LAMAR_LIN_SCENES = sorted([p.name for p in LAMAR_LIN_DATA_PATH.glob("ios*") if p.is_dir()])

METHODS = ['COLMAP', 'GLOMAP', 'VGGSfM', 'FlowMap', 'AceZero']

# Results Path
RESULTS_PATH = Path(path / "data/results/").absolute()
COLMAP_RESULTS_PATH = RESULTS_PATH / "colmap"
GLOMAP_RESULTS_PATH = RESULTS_PATH / "glomap"
FLOWMAP_RESULTS_PATH = RESULTS_PATH / "flowmap"
VGGSFM_RESULTS_PATH = RESULTS_PATH / "vggsfm"
ACEZERO_RESULTS_PATH = RESULTS_PATH / "acezero"
COLMAP_FORMAT = "colmap/sparse/0"
COLMAP_DENSE_FORMAT = "colmap/dense/"

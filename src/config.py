from pathlib import Path

# Data Path
ETH3D_DATA_PATH = Path("../data/datasets/ETH3D")
MIP_NERF_360_DATA_PATH = Path("../data/datasets/MipNerf360")
TANKS_AND_TEMPLES_DATA_PATH = Path("../data/datasets/TanksAndTemples")

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

METHODS = ['Glomap', 'VGGSfm', 'FlowMap', 'AceZero']

# Results Path
RESULTS_PATH = Path("../data/results/")
GLOMAP_RESULTS_PATH = RESULTS_PATH / "glomap"
FLOWMAP_RESULTS_PATH = RESULTS_PATH / "flowmap"
VGGSFM_RESULTS_PATH = RESULTS_PATH / "vggsfm"
ACEZERO_RESULTS_PATH = RESULTS_PATH / "acezero"
COLMAP_FORMAT = "colmap/sparse/0"
COLMAP_DENSE_FORMAT = "colmap/dense/"

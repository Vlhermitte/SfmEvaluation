from pathlib import Path

# Path to the ETH3D Multi-view evaluation tool
MULTIVIEW_EVALUATION_TOOL_PATH = Path("../multi-view-evaluation/build/ETH3DMultiViewEvaluation")

# Data Path
ETH3D_DATA_PATH = Path("../data/datasets/ETH3D")
MIP_NERF_360_DATA_PATH = Path("../data/datasets/MipNerf360")

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

METHODS = ['Glomap', 'VGGSfm', 'FlowMap', 'AceZero']

# Results Path
RESULTS_PATH = Path("../data/results/")
GLOMAP_RESULTS_PATH = RESULTS_PATH / "glomap"
FLOWMAP_RESULTS_PATH = RESULTS_PATH / "flowmap"
VGGSFM_RESULTS_PATH = RESULTS_PATH / "vggsfm"
ACEZERO_RESULTS_PATH = RESULTS_PATH / "acezero"
COLMAP_FORMAT = "colmap/sparse/0"

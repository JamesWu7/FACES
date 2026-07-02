from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BINARY_MODEL_DIR = PROJECT_ROOT / "Binary classification model"
SUPERCLASS_MODEL_DIR = PROJECT_ROOT / "three-category superclass classification model"
SUBTYPE_MODEL_DIR = PROJECT_ROOT / "11-class ResNet50 subtype model"

BINARY_WEIGHTS = BINARY_MODEL_DIR / "best_auc_model_seed64.pth"
SUPERCLASS_WEIGHTS = SUPERCLASS_MODEL_DIR / "best_auc_model_seed53.pth"
SUBTYPE_WEIGHTS = SUBTYPE_MODEL_DIR / "best.pth"
DLIB_LANDMARK_MODEL = BINARY_MODEL_DIR / "shape_predictor_68_face_landmarks.dat"

BINARY_CLASSES = ["无病(HC)", "有病"]
SUPERCLASS_CLASSES = ["SS", "CMS", "AIS"]
SUPERCLASS_FULL_NAMES = {
    "SS": "SS (Syndromic Scoliosis)",
    "CMS": "CMS (Chiari malformation-associated scoliosis)",
    "AIS": "AIS (Adolescent Idiopathic Scoliosis)",
}

SS_SUBTYPE_CLASSES = [
    "AMC (Arthrogryposis Multiplex Congenita)",
    "EDS (Ehlers–Danlos Syndrome)",
    "FSS (Freeman-Sheldon syndrome)",
    "GSD (Gorham-Stout disease)",
    "MFS (Marfan syndrome)",
    "NF-1 (Neurofibromatosis type 1)",
    "Osteochondrodysplasia",
    "Osteogenesis imperfecta",
    "Other Syndrome",
    "PWS (Prader-Willi syndrome)",
    "SGS (Shprintzen-Goldberg syndrome)",
]

SS_DISPLAY_THRESHOLD = 0.50

MODEL_VERSIONS = {
    "binary": BINARY_WEIGHTS.name,
    "etiology": SUPERCLASS_WEIGHTS.name,
    "ss_subtype": SUBTYPE_WEIGHTS.name,
}

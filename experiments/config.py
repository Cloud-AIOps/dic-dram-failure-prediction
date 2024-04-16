

TABLES = [
    "aer_event_alerts",
    "bmc_trap_mem_alerts",
    "compute_node_static_infos",
    "compute_node_status_infos",
    "cpu_topology_infos",
    "edac_topology_infos",
    "extlog_event_alerts",
    "mc_event_alerts",
    "mce_record_alerts",
    "memory_topology_infos"
]

PATH_RAW = "../data/raw"
PATH_TMP = "../data/tmp"
PATH_TRAIN = "../data/train"
PATH_ASSETS = "../data/assets"
PATH_RESULT = "../results"
PATH_FEATURES = "../data/features"
PATH_PROCESSED = "../data/processed"
PATH_PROCESSED_AGG = "../data/processed/agg"
PATH_MODEL = "../data/model"

DB_FILE_TEMPLATE = PATH_ASSETS + "/template.sqlite"
DB_FILE_LATEST = PATH_PROCESSED + "/latest-ckpt.sqlite"


START_DATE = "2023-01-01"
END_DATE = "2023-07-30"

# version
MODEL_VERSION = "ConvRule-model-beta-0.1.pth"
MODEL_VERSION_TEST = "ConvRule-model-test-0.1.pth"

# common variables
DRAM_MODEL_MAPS = {"A1": 0.0, "A2": 1.0, "B1": 2.0, "B2": 3.0, "B3": 4.0, "C1": 5.0, "C2": 6.0}
DRAM_MODEL_SCALING_PARAM = 10000000
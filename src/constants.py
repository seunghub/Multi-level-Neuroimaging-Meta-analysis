from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
CACHE_PATH = PROJECT_PATH / ".cache"
CACHE_PATH.mkdir(exist_ok=True)
DATA_PATH = PROJECT_PATH / "data"
DATA_PATH.mkdir(exist_ok=True)
FMRI_DECODING_PATH = Path("/data/parietal/store2/work/rmeudec/large-scale-fmri-decoding")
FMRI_DATA_PATH = FMRI_DECODING_PATH / "data"

from datasets import load_dataset

dataset = load_dataset(
    "google/fleurs",
    "ms_my",
    cache_dir="/mnt/weka/aisg/speech_spoke/donghang/hf_cache/datasets",
)
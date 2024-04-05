import os
from pydoc import locate

from jsonargparse import ActionConfigFile, ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    parser = ArgumentParser()
    # Main args
    parser.add_argument("--model_name", "-m", default="xlm-roberta-large")
    parser.add_argument("--data", "-d", default="")
    parser.add_argument("--language", "-l", default="sv")

    cfg = parser.parse_args()
    print(parser.dump(cfg))

    locate(f"lineclassifier5.predict").run(cfg)

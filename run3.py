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
    parser.add_argument("--method", "-me", default="train")
    parser.add_argument("--model_name", "-m", default="xlm-roberta-large")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--sample", "-sa", type=int, default=0)
    parser.add_argument("--languages", "-l", default="de-en-es-fi-fr-se")
    parser.add_argument("--splits", default="train-dev-test")

    # Trainer
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", "-bt", type=int, default=8)
    parser.add_argument("--eval_batch_size", "-bd", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda")

    cfg = parser.parse_args()
    print(parser.dump(cfg))

    method = cfg.method if cfg.method != "test" else "train"

    locate(f"lineclassifier3.{method}").run(cfg)

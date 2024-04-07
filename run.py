import os
from pydoc import locate

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", "-m", default="train")
    parser.add_argument("--base_model_name", default="BAAI/bge-m3")
    parser.add_argument("--trained_model_name", default="models/base")
    parser.add_argument("--lstm_model_name", default="models/lstm/best_model.pt")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--languages", default="de-en-es-fi-fr-se")
    parser.add_argument("--predict_language", default="sv")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base_learning_rate", type=float, default=1e-5)
    parser.add_argument("--lstm_learning_rate", type=float, default=5e-6)
    parser.add_argument("--embeddings_file", default="embeddings.pkl")

    cfg = parser.parse_args()
    print(parser.dump(cfg))

    locate(f"lineclassifier.{cfg.method}").run(cfg)

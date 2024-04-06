import os
from pydoc import locate

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    parser = ArgumentParser()
    # Main args
    parser.add_argument("--method", "-me", default="train")
    parser.add_argument(
        "--embedding_file", default="documents_embeddings_with_labels_bge-m3.pkl"
    )
    parser.add_argument("--model_name", "-m", default="xlm-roberta-large")
    parser.add_argument("--model_type", default="lstm")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--patience", "-p", type=int, default=3)

    # Trainer
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", "-bt", type=int, default=8)
    parser.add_argument("--eval_batch_size", "-bd", type=int, default=8)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda")

    cfg = parser.parse_args()
    print(parser.dump(cfg))

    method = cfg.method if cfg.method != "test" else "train"

    locate(f"lstm_model.{method}").run(cfg)

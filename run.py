import os

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--language", default="fi")
    parser.add_argument("--model", default="xlm-roberta-large")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--b", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pool", type=bool, default=True)

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from lineclassifier import main

    main.run(cfg)

import os

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--language", "-l", default="fi")
    parser.add_argument("--model", "-m", default="xlm-roberta-large")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--eval_strategy", default="epoch")
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--context", type=int, default=2048)

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from lineclassifier import main

    main.run(cfg)

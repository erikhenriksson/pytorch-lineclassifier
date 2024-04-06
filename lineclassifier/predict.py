import json
import zstandard as zstd
import io

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch
from dotenv import dotenv_values
from datasets import load_dataset
from huggingface_hub import login


def stream_zst_json_lines(file_path, lang):
    dctx = zstd.ZstdDecompressor()
    with open(file_path, "rb") as compressed_file:
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for doc in text_stream:
                doc_parsed = json.loads(doc)
                data = []
                for line, label in zip(
                    doc_parsed["content"].split("\n"),
                    doc_parsed["metadata"]["sentence_identifications"],
                ):
                    if label and label["label"] == lang:
                        data.append(line)

                yield data, doc_parsed["warc_headers"]["warc-record-id"]


def parse_and_clean(doc, lang):
    data = []
    for line, label in zip(
        doc["content"].split("\n"),
        doc["metadata"]["sentence_identifications"],
    ):
        if label and label["label"] == lang:
            data.append(line)

    return data, doc["warc_headers"]["warc-record-id"]


def run(cfg):
    env = dotenv_values(".env")
    login(token=env["HF_TOKEN"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        cfg.predict_language,
        "train",
        streaming=True,
        trust_remote_code=True,
    )
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10)
    for ex in shuffled_dataset["train"]:
        print(ex)
        exit()

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    tuned_model = XLMRobertaForSequenceClassification.from_pretrained(
        cfg.model_name
    ).to(device)

    exit()

    # Example usage
    file_path = cfg.data
    i = 0

    for json_object in stream_zst_json_lines(file_path, cfg.language):

        lines, rec_id = json_object
        non_junk_lines = []

        for line in lines:
            inputs = tokenizer(
                line,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = tuned_model(**inputs)

                predictions = torch.argmax(outputs.logits, dim=-1)

                is_not_junk = (
                    predictions.item() == 1
                )  # Adjust label index according to your model

                if is_not_junk:
                    non_junk_lines.append(line)

        result_text = " ".join(non_junk_lines)
        out_file = cfg.data.split("/")[-1].split(".zst")[0]
        if result_text:
            with open(f"cleaned/{out_file}", "a", encoding="utf-8") as file:
                file.write(
                    json.dumps({"id": rec_id, "text": result_text}, ensure_ascii=False)
                    + "\n"
                )

        i += 1

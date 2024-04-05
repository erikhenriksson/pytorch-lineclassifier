import json
import zstandard as zstd
import io

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch


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

                yield data


def run(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    tuned_model = XLMRobertaForSequenceClassification.from_pretrained(
        cfg.model_name
    ).to(device)

    # Example usage
    file_path = cfg.data
    for json_object in stream_zst_json_lines(file_path, "sv"):

        lines = json_object
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

        result_text = "\n".join(non_junk_lines)
        print(result_text)
        exit()
        return result_text

        # FOR EACH LINE, USE THE MODEL TO PREDICT THE LABEL (JUNK OR NOT JUNK)

        # THEN, CONCATENATE THE LINES THAT ARE NOT JUNK WITH "\N"

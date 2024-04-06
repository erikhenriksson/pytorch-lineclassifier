import torch
import pickle
import json
from transformers import (
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    AutoTokenizer,
)
from tqdm import tqdm


def lines_to_embeddings(document, tokenizer, model, device):

    # Batch the lines for efficiency

    all_data = []  # To store the final embeddings and labels
    all_lines = document["text"]
    all_labels = document["labels"]
    batch_size = 128

    # Process in batches
    for i in range(0, len(all_lines), batch_size):
        batch_lines = all_lines[i : i + batch_size]
        batch_labels = all_labels[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[
                :, 0, :
            ]  # Extract the [CLS] token's embeddings for each line
            cls_embeddings = (
                cls_embeddings.cpu().numpy()
            )  # Move embeddings to CPU and convert to numpy for easier handling

            batch_data = list(zip(cls_embeddings, batch_labels))
            all_data.extend(batch_data)

    return all_data


def run(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model = XLMRobertaModel.from_pretrained(cfg.model_name).to(device)

    # Let's get the original model's name
    with open(f"{cfg.model_name}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path", None))

    # Load documents
    documents = {"train": [], "dev": [], "test": []}
    for language in cfg.languages.split("-"):
        for split in ["train", "dev", "test"]:
            with open(f"data/{language}/{split}.json", "r", encoding="utf-8") as file:
                data = json.load(file)
            for item in data:
                documents[split].append(item)

    # Convert each line in the documents to embeddings
    for split in documents:

        for i, document in enumerate(
            tqdm(documents[split], desc=f"Processing {split}")
        ):
            embeddings = lines_to_embeddings(document, tokenizer, model, device)
            # Replace the text in the document with its embeddings
            documents[split][i] = embeddings

    pickle_file = "documents_embeddings_with_labels.pkl"

    with open(pickle_file, "wb") as f:
        pickle.dump(documents, f)

    print(f"Embeddings and labels saved to {pickle_file}")

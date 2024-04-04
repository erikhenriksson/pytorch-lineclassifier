import torch
import pickle
import json
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from tqdm import tqdm


def lines_to_embeddings(document, tokenizer, model, device):
    """
    Convert lines in a document to embeddings.

    Args:
    - document: A dictionary with a "text" key holding a list of lines.
    - tokenizer: An instance of XLMRobertaTokenizer.
    - model: An instance of XLMRobertaModel.
    - device: The torch device to run the model on.

    Returns:
    - A list of 1024-dimensional embeddings, one for each line in the document.
    """
    # Batch the lines for efficiency

    embeddings = []
    with torch.no_grad():  # No need to compute gradients
        for line in document["text"]:
            inputs = tokenizer(
                line,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # outputs = model(**inputs)
            # Use the average of the last hidden states as the line's embedding
            # last_hidden_states = outputs.last_hidden_state
            # mean_embedding = torch.mean(last_hidden_states, dim=1)
            with torch.no_grad():  # Ensure no gradients are computed for memory efficiency
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze().cpu().numpy())

    return embeddings


def run(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaForSequenceClassification.from_pretrained(cfg.model_name).to(
        device
    )

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
            documents[split][i]["text"] = embeddings

    # Assuming 'documents' is your dictionary containing the embeddings
    pickle_file = "documents_embeddings.pkl"

    # Saving the embeddings to a file
    with open(pickle_file, "wb") as f:
        pickle.dump(documents, f)

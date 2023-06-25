import torch


def label_int2str(row, dataset):
    """
    Convert integer label to string label according to the dataset.
    Example:
        banking = load_dataset("banking77")
        banking.set_format(type='pandas')
        train_df = banking['train'][:]
        train_df['label_name'] = train_df['label'].apply(label_int2str, args=(banking['train'],))
    """
    return dataset.features["label"].int2str(row)


def tokenize(batch, tokenizer):
    """
    Tokenize the text using the tokenizer.
    Example:
        banking = load_dataset("banking77")
        tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
        encoded_train = banking["train"].map(
            tokenize, batched=True, batch_size=None, fn_kwargs={"tokenizer": tokenizer}
        )
    """
    return tokenizer(batch["text"], padding=True, truncation=True)


def extract_hidden_states(batch, model, tokenizer, device="cpu"):
    """
    Extract the hidden states from the encoder model.
    Example:
        banking = load_dataset("banking77")
        tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
        encoded_banking = banking.map(
            tokenize, batched=True, batch_size=None, fn_kwargs={"tokenizer": tokenizer}
        )
        encoded_hidden = encoded_banking.map(extract_hidden_states, batched=True, fn_kwargs={'model': model, 'tokenizer': tokenizer, 'device': device})
    """
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

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

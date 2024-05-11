import os
from torch.utils.data import Dataset
from transformers import BertTokenizer

class PTBDataset(Dataset):
    """
    Penn Treebank Dataset for training Sentence Variational Autoencoder models.
    This dataset loads the Penn Treebank (PTB) data files, preprocesses text lines
    using a BERT tokenizer, and creates the necessary inputs for the SentenceVAE model.
    """
    def __init__(self, data_dir, split, max_sequence_length=128):
        """
        Initializes the PTBDataset.

        Args:
            data_dir (str): Directory where the data files are located.
            split (str): Type of dataset split ('train', 'valid', or 'test').
            max_sequence_length (int): Maximum length of tokenized sequences.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sos_token_id = self.tokenizer.convert_tokens_to_ids('[unused0]')
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('[unused1]')
        self.pad_token_id = self.tokenizer.pad_token_id
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads and preprocesses the data from files.

        Returns:
            List[Dict]: List of dictionaries containing the input_ids, attention_mask, and target_ids.
        """
        file_path = os.path.join(self.data_dir, f'ptb.{self.split}.txt')
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                modified_line = '[unused0] ' + line.strip() + ' [unused1]'
                tokens = self.tokenizer.encode_plus(
                    modified_line,
                    add_special_tokens=True,
                    max_length=self.max_sequence_length,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                data.append({
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'target_ids': tokens['input_ids'].squeeze(0)  # Target is same as input for autoencoding tasks
                })
        return data

    def __len__(self):
        """
        Returns the number of examples in this dataset.

        Returns:
            int: Total number of examples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict: A dictionary containing the input_ids, attention_mask, and target_ids.
        """
        return self.data[idx]

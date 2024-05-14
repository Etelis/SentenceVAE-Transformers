from torch.utils.data import Dataset
from transformers import BartTokenizer
import os

class PTB(Dataset):
    """
    Penn Treebank (PTB) Dataset for Language Modeling using BART Tokenizer.

    Args:
        data_dir (str): Directory containing the PTB dataset files.
        split (str): Data split to load ('train', 'valid', or 'test').
        max_sequence_length (int, optional): Maximum length of sequences for input to the model. Defaults to 128.
    """
    def __init__(self, data_dir, split, max_sequence_length=128):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads and tokenizes the PTB data from the specified split.

        Returns:
            list of dict: A list of dictionaries containing tokenized input_ids, attention_mask, and target_ids.
        """
        file_path = os.path.join(self.data_dir, f'ptb.{self.split}.txt')
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                tokens = self.tokenizer.encode_plus(
                    line,
                    add_special_tokens=True,
                    max_length=self.max_sequence_length,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                target_ids = tokens['input_ids'].clone()
                target_ids[:, :-1] = tokens['input_ids'][:, 1:]
                target_ids[:, -1] = self.tokenizer.pad_token_id

                data.append({
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'target_ids': target_ids.squeeze(0),
                })

        return data

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and target_ids for the sample.
        """
        return self.data[idx]

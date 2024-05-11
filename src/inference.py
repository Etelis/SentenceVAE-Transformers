import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from dataset import PTB
from utils import load_model

def generate_sentences(model, dataloader, num_samples, tokenizer, device):
    """
    Generates a specified number of sentences from a trained model using a given dataloader.

    Args:
        model (torch.nn.Module): The trained SentenceVAE model to generate sentences.
        dataloader (DataLoader): DataLoader that provides batches of input data for sentence generation.
        num_samples (int): Number of sentences to generate.
        tokenizer (BertTokenizer): Tokenizer used to decode the generated token IDs back into text.
        device (torch.device): Device on which the model and data should be processed (e.g., 'cuda' or 'cpu').

    Returns:
        list of str: A list containing the generated sentences.
    """
    model.eval()
    samples_collected = 0
    sentences = []
    for batch in dataloader:
        if samples_collected >= num_samples:
            break
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        with torch.no_grad():
            logits_sequence, _, _ = model(input_ids, attention_mask)

        predicted_token_ids = torch.argmax(logits_sequence, dim=-1)
        decoded_sentences = tokenizer.batch_decode(predicted_token_ids.tolist(), skip_special_tokens=True)

        for sentence in decoded_sentences:
            if samples_collected < num_samples:
                sentences.append(sentence)
                samples_collected += 1
            else:
                break

    return sentences

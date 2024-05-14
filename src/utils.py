import torch
import os
import json
import numpy as np
from model import SentenceVAE

def kl_anneal_function(anneal_function, step, k, annealing_till):
    """
    Computes the annealing factor for the KL divergence.

    Args:
        anneal_function (str): The type of annealing function to use ('logistic' or 'linear').
        step (int): The current training step.
        k (float): The annealing rate.
        annealing_till (int): The step until which to anneal.

    Returns:
        float: The annealing factor.
    """
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - annealing_till))))
    elif anneal_function == 'linear':
        return min(1, step / annealing_till)

def save_model_and_config(args, model, timestamp):
    """
    Saves the trained model and its configuration.

    Args:
        args (Namespace): Arguments containing configuration for saving the model.
        model (SentenceVAE): The trained model.
        timestamp (str): Timestamp to append to the model file names.
    """
    model_path = os.path.join(args.save_model_path, f"model_final_{timestamp}.pt")
    config_path = os.path.join(args.save_model_path, f"model_final_{timestamp}_config.json")
    
    # Save model state_dict
    torch.save(model.state_dict(), model_path)
    print(f"Model state dict saved to {model_path}")

    # Save model config
    model_config = {
        "latent_dim": model.latent_dim,
        "hidden_size": model.hidden_size,
        "vocab_size": model.vocab_size,
        "special_tokens": model.special_tokens,
        "max_seq_length": model.max_seq_length,
        "word_dropout_rate": model.word_dropout_rate,
        "gru_layers": model.gru_layers,
    }
    
    with open(config_path, 'w') as f:
        json.dump(model_config, f)
    print(f"Model config saved to {config_path}")

def load_model(config_path, model_path, device):
    """
    Loads a trained SentenceVAE model from a configuration file and checkpoint.

    Args:
        config_path (str): Path to the model configuration file.
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model onto.

    Returns:
        SentenceVAE: Loaded model.
    """
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    model = SentenceVAE(
        latent_dim=model_config['latent_dim'],
        hidden_size=model_config['hidden_size'],
        vocab_size=model_config['vocab_size'],
        special_tokens=model_config['special_tokens'],
        max_seq_length=model_config['max_seq_length'],
        word_dropout_rate=model_config['word_dropout_rate'],
        gru_layers=model_config['gru_layers']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

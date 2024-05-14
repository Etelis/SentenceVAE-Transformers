import os
import torch
from transformers import BartTokenizer
from utils import load_model

def generate_sentences(model, tokenizer, num_samples, beam_width=5):
    """
    Generate sentences using the beam search method.

    Args:
        model (SentenceVAE): Trained SentenceVAE model.
        tokenizer (BartTokenizer): Tokenizer for decoding the generated token IDs.
        num_samples (int): Number of sentences to generate.
        beam_width (int, optional): Beam width for the search. Defaults to 5.

    Returns:
        list: Generated sentences.
    """
    model.eval()
    sentences = []
    for _ in range(num_samples):
        z = torch.randn((1, model.latent_dim), device=model.device)
        generated_ids = model.beam_search(z, beam_width=beam_width)
        generated_sentence = tokenizer.decode(generated_ids, skip_special_tokens=True)
        sentences.append(generated_sentence)
    return sentences

def inference(args):
    """
    Perform inference using a trained SentenceVAE model.

    Args:
        args (Namespace): Arguments containing configuration for inference.
    """
    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(f"Model checkpoint file not found: {args.load_checkpoint}")

    config_path = args.load_checkpoint.replace('.pt', '_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = load_model(config_path, args.load_checkpoint, args.device)
    model.eval()

    samples = generate_sentences(model, tokenizer, args.num_samples, args.beam_width)
    print('----------GENERATED SAMPLES----------')
    print(*samples, sep='\n')

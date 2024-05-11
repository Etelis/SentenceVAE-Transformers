import argparse
import subprocess
from train import setup_datasets_and_loaders, setup_model, train, validate
from inference import generate_sentences
from utils import load_model, plot_metrics
from dataset import PTB
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch

def main():
    parser = argparse.ArgumentParser(description="Train or run inference using the SentenceVAE model.")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help='Operational mode: "train" or "inference"')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--gru_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_model_path', type=str, default='bin')
    parser.add_argument('--load_checkpoint', type=str, help='Path to the model checkpoint for inference')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate in inference mode')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data before starting training or inference
    print("Preparing data...")
    subprocess.call(['bash', './prepare_data.sh'])

    if args.mode == 'train':
        datasets, dataloaders = setup_datasets_and_loaders(args)
        model = setup_model(args, datasets)
        model.to(device)
        # Assume training function includes necessary training loops
    elif args.mode == 'inference':
        if not args.load_checkpoint:
            raise ValueError("Checkpoint path must be provided for inference mode.")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = load_model(args.load_checkpoint, device)
        dataset = PTB(args.data_dir, 'test', args.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        sentences = generate_sentences(model, dataloader, args.num_samples, tokenizer, device)
        for sentence in sentences:
            print("Generated sentence:", sentence)

if __name__ == '__main__':
    main()

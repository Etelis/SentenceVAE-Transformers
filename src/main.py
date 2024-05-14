import argparse
import subprocess
from train import train_model
from inference import inference
import torch

def main():
    parser = argparse.ArgumentParser(description="Train or run inference using the SentenceVAE model.")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help='Operational mode: "train" or "inference"')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and inference')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension size for VAE')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for GRU layers')
    parser.add_argument('--gru_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--save_model_path', type=str, default='bin', help='Path to save the trained model')
    parser.add_argument('--load_checkpoint', type=str, help='Path to the model checkpoint for inference')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate in inference mode')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search in inference mode')
    parser.add_argument('--word_dropout_rate', type=float, default=0.2, help='Word dropout rate for training')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Preparing data...")
    subprocess.call(['bash', './prepare_data.sh'])

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'inference':
        if not args.load_checkpoint:
            raise ValueError("Checkpoint path must be provided for inference mode.")
        inference(args)

if __name__ == '__main__':
    main()

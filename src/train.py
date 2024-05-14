import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from model import SentenceVAE
from dataset import PTB
from utils import kl_anneal_function, save_model_and_config

def prepare_datasets_and_loaders(args):
    """
    Prepare the datasets and dataloaders for training.

    Args:
        args (Namespace): Arguments containing configuration for the datasets and dataloaders.

    Returns:
        dict: Datasets for training.
        dict: Dataloaders for training.
    """
    datasets = {
        'train': PTB(args.data_dir, 'train', args.max_seq_length)
    }
    if args.test:
        datasets['test'] = PTB(args.data_dir, 'test', args.max_seq_length)

    dataloaders = {
        x: DataLoader(
            datasets[x], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        ) 
        for x in datasets
    }
    return datasets, dataloaders

def initialize_model(args, datasets):
    """
    Initialize the SentenceVAE model with the appropriate configuration.

    Args:
        args (Namespace): Arguments containing model configuration.
        datasets (dict): Datasets to extract tokenizer and special tokens information.

    Returns:
        SentenceVAE: Initialized model.
    """
    special_tokens = {
        'bos_token_id': datasets['train'].tokenizer.bos_token_id,
        'eos_token_id': datasets['train'].tokenizer.eos_token_id,
        'pad_token_id': datasets['train'].tokenizer.pad_token_id,
        'unk_token_id': datasets['train'].tokenizer.unk_token_id
    }
    
    return SentenceVAE(
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        vocab_size=len(datasets['train'].tokenizer),
        special_tokens=special_tokens,
        max_seq_length=args.max_seq_length,
        word_dropout_rate=args.word_dropout_rate,
        gru_layers=args.gru_layers
    )

def configure_loss_function(datasets):
    """
    Configure the loss function for training.

    Args:
        datasets (dict): Datasets to extract the tokenizer's pad token ID.

    Returns:
        torch.nn.NLLLoss: Configured negative log-likelihood loss function.
    """
    return torch.nn.NLLLoss(ignore_index=datasets['train'].tokenizer.pad_token_id, reduction='sum')

def train_epoch(model, dataloader, loss_fn, optimizer, epoch, args, writer):
    """
    Train the model for one epoch.

    Args:
        model (SentenceVAE): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch number.
        args (Namespace): Arguments containing configuration for training.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
    """
    model.train()
    total_steps = len(dataloader.dataset) * epoch
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} train")

    for batch_idx, batch in progress_bar:
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target_ids']
        input_ids, attention_mask, targets = input_ids.to(args.device), attention_mask.to(args.device), targets.to(args.device)

        logits, mu, log_sigma_squared, z = model(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(1, 2)
        nll_loss = loss_fn(log_probs, targets)

        current_step = total_steps + batch_idx

        kl_weight = kl_anneal_function(args.anneal_function, current_step, args.k, args.annealing_till)
        kl_loss = -0.5 * torch.sum(1 + log_sigma_squared - mu.pow(2) - log_sigma_squared.exp()) / input_ids.size(0)
        weighted_kl_loss = kl_weight * kl_loss

        loss = nll_loss + weighted_kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        writer.add_scalar("train/KL Divergence", kl_loss.item(), current_step)
        writer.add_scalar("train/Negative Log-Likelihood", nll_loss.item(), current_step)
        writer.add_scalar("train/ELBO", -loss.item(), current_step)

        progress_bar.set_postfix({'nll_loss': nll_loss.item(), 'kl_loss': kl_loss.item(), 'total_loss': loss.item()})

def train_model(args):
    """
    Main function to train the model.

    Args:
        args (Namespace): Arguments containing configuration for training.
    """
    writer = SummaryWriter()

    datasets, dataloaders = prepare_datasets_and_loaders(args)
    model = initialize_model(args, datasets)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = configure_loss_function(datasets)

    for epoch in range(args.epochs):
        train_epoch(model, dataloaders['train'], loss_fn, optimizer, epoch, args, writer)

    save_model_and_config(args, model, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    writer.close()
    print("Training completed and model saved. All data logged to TensorBoard.")

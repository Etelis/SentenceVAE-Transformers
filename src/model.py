import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BartModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentenceVAE(nn.Module):
    """
    Sentence Variational Autoencoder (VAE) with BART Encoder and GRU Decoder.

    Args:
        latent_dim (int): Dimension of the latent space.
        hidden_size (int): Size of the GRU hidden state.
        vocab_size (int): Size of the vocabulary.
        special_tokens (dict): Dictionary of special tokens (e.g., bos_token_id, eos_token_id, pad_token_id, unk_token_id).
        max_seq_length (int): Maximum sequence length.
        word_dropout_rate (float): Word dropout rate for the decoder input.
        gru_layers (int, optional): Number of GRU layers. Defaults to 1.
        freeze_bart (bool, optional): Whether to freeze BART encoder parameters. Defaults to True.
    """
    def __init__(self, latent_dim, hidden_size, vocab_size, special_tokens, max_seq_length, word_dropout_rate, gru_layers=1, freeze_bart=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.special_tokens = special_tokens
        self.gru_layers = gru_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_dropout_rate = word_dropout_rate

        self.bart_encoder = BartModel.from_pretrained('facebook/bart-base').encoder.to(self.device)
        self.bart_embeddings = self.bart_encoder.embed_tokens

        if freeze_bart:
            for param in self.bart_encoder.parameters():
                param.requires_grad = False
            for param in self.bart_embeddings.parameters():
                param.requires_grad = True

        self.encoder_to_latent_mu = nn.Linear(self.bart_encoder.config.hidden_size, latent_dim).to(self.device)
        self.encoder_to_latent_logsigma = nn.Linear(self.bart_encoder.config.hidden_size, latent_dim).to(self.device)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * self.gru_layers).to(self.device)
        self.decoder = nn.GRU(self.bart_encoder.config.hidden_size, hidden_size, num_layers=gru_layers, batch_first=True).to(self.device)
        self.output_to_vocab = nn.Linear(hidden_size, vocab_size).to(self.device)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the SentenceVAE model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input IDs.

        Returns:
            logits (torch.Tensor): Logits for the vocabulary.
            mu (torch.Tensor): Mean of the latent space distribution.
            log_sigma_squared (torch.Tensor): Log variance of the latent space distribution.
            z (torch.Tensor): Sampled latent vector.
        """
        # Encoder
        encoder_outputs = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_last_hidden_state = encoder_outputs.last_hidden_state.mean(dim=1)

        # Latent space
        mu = self.encoder_to_latent_mu(encoder_last_hidden_state)
        log_sigma_squared = self.encoder_to_latent_logsigma(encoder_last_hidden_state)
        sigma = torch.exp(0.5 * log_sigma_squared)
        z = mu + sigma * self.N.sample(mu.shape)

        # Remove EOS token from sequences
        eos_token_id = self.special_tokens['eos_token_id']
        pad_token_id = self.special_tokens['pad_token_id']
        input_ids_no_eos = input_ids.clone()
        input_ids_no_eos[input_ids == eos_token_id] = pad_token_id

        # Preparing the decoder's inputs with word dropout
        if self.word_dropout_rate > 0:
            prob = torch.rand(input_ids.shape, device=self.device)
            mask = (prob < self.word_dropout_rate) & (input_ids_no_eos != pad_token_id)
            input_ids_no_eos = input_ids_no_eos.masked_fill(mask, self.special_tokens['unk_token_id'])

        decoder_input_ids = input_ids_no_eos
        decoder_input = self.bart_embeddings(decoder_input_ids)

        # Calculate lengths for packed sequence
        lengths = attention_mask.sum(dim=1).cpu() - 1

        # Pack the sequences
        packed_input = pack_padded_sequence(decoder_input, lengths, batch_first=True, enforce_sorted=False)

        # Decoder
        decoder_hidden = self.latent_to_hidden(z).view(self.gru_layers, -1, self.hidden_size)
        packed_output, _ = self.decoder(packed_input, decoder_hidden)
        decoder_outputs, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.max_seq_length)

        # Output logits
        logits = self.output_to_vocab(decoder_outputs)

        return logits, mu, log_sigma_squared, z

    def beam_search(self, z, beam_width=5, max_seq_length=128):
        """
        Beam search for sequence generation.

        Args:
            z (torch.Tensor): Latent vector.
            beam_width (int, optional): Beam width for the search. Defaults to 5.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 128.

        Returns:
            list: Generated token IDs.
        """
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z).view(self.gru_layers, batch_size, self.hidden_size)

        start_token_id = self.special_tokens['bos_token_id']
        end_token_id = self.special_tokens['eos_token_id']

        sequences = [[[start_token_id], 0.0]]  # Each sequence is (tokens, score)

        for _ in range(max_seq_length):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == end_token_id:
                    all_candidates.append((seq, score))
                    continue

                input_sequence = torch.tensor([seq[-1]], device=self.device).unsqueeze(0)
                input_embedding = self.bart_embeddings(input_sequence)
                decoder_output, hidden = self.decoder(input_embedding, hidden)  # No need to unsqueeze input_embedding here
                logits = self.output_to_vocab(decoder_output.squeeze(1))
                log_probs = F.log_softmax(logits, dim=-1)

                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    candidate = (seq + [topk_tokens[0, i].item()], score + topk_log_probs[0, i].item())
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]

            # Stop if all sequences end with the end token
            if all(seq[-1] == end_token_id for seq, _ in sequences):
                break

        return sequences[0][0]
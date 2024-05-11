import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = 'bert-base-uncased'

class SentenceVAE(nn.Module):
    """A Sentence Variational Autoencoder utilizing BERT as encoder and GRU as decoder.

    Attributes:
        latent_dim (int): Dimensionality of the latent space.
        hidden_size (int): Size of the hidden layers for GRU.
        vocab_size (int): Size of the vocabulary.
        max_seq_length (int): Maximum sequence length for inputs and outputs.
        gru_layers (int): Number of GRU layers.
        special_tokens (dict): Dictionary of special tokens used in decoding.
    """

    def __init__(self, latent_dim, hidden_size, vocab_size, special_tokens, max_seq_length, gru_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.special_tokens = special_tokens
        self.gru_layers = gru_layers

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.encoder_to_latent_mu = nn.Linear(self.bert.config.hidden_size, latent_dim)
        self.encoder_to_latent_logsigma = nn.Linear(self.bert.config.hidden_size, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * self.gru_layers)

        self.embedding = nn.Embedding(vocab_size, self.bert.config.hidden_size)
        self.decoder = nn.GRU(self.bert.config.hidden_size, hidden_size, num_layers=gru_layers, batch_first=True)
        self.output_to_vocab = nn.Linear(hidden_size, vocab_size)

        self.normal_distribution = torch.distributions.Normal(0, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model.

        Args:
            input_ids (Tensor): Input tensor containing token IDs.
            attention_mask (Tensor): Mask to avoid processing of padding tokens.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing the logits sequence, mu, and log sigma squared.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoder_last_hidden_state = outputs.last_hidden_state[:, 0]

        mu = self.encoder_to_latent_mu(encoder_last_hidden_state)
        log_sigma_squared = self.encoder_to_latent_logsigma(encoder_last_hidden_state)
        sigma = torch.exp(0.5 * log_sigma_squared)
        z = mu + sigma * self.normal_distribution.sample(mu.shape).to(device)

        decoder_hidden = self.latent_to_hidden(z).view(self.gru_layers, -1, self.hidden_size)

        initial_token = torch.full((input_ids.size(0), 1), self.special_tokens['sos_token'], dtype=torch.long, device=device)
        decoder_input = self.embedding(initial_token)
        logits_list = []

        for step in range(self.max_seq_length):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            logits = self.output_to_vocab(output.squeeze(1))
            logits_list.append(logits.unsqueeze(1))

            probabilities = F.softmax(logits, dim=-1)
            _, next_token = torch.max(probabilities, dim=-1)
            decoder_input = self.embedding(next_token.unsqueeze(1))

        logits_sequence = torch.cat(logits_list, dim=1)

        return logits_sequence, mu, log_sigma_squared

    def generate(self, num_samples):
        """Generates sentences from the latent space without input.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            Tensor: Generated token IDs.
        """
        z = torch.randn((num_samples, self.latent_dim), device=device)
        decoder_hidden = z.unsqueeze(0)
        outputs = torch.full((num_samples, 1), self.special_tokens['sos_token'], dtype=torch.long, device=device)

        for _ in range(self.max_seq_length):
            output, decoder_hidden = self.decoder(outputs[:, -1:], decoder_hidden)
            logits = self.output_to_vocab(output.squeeze(1))
            _, next_token = torch.max(logits, dim=-1)
            outputs = torch.cat((outputs, next_token.unsqueeze(1)), dim=1)

        return outputs

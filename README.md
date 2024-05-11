Sentence VAE: Generating Sentences from a Continuous Space
==========================================================

Overview
--------

This repository contains a reimplementation of the approach described in the paper "Generating Sentences from a Continuous Space." The paper introduces a method for modeling sentences as latent space representations, which enables the deterministic decoding of these vectors to produce grammatically correct and semantically coherent sentences. Additionally, the model facilitates the interpolation between two latent vectors, resulting in coherent sentence transitions.

Inspiration
-----------

This implementation draws inspiration from Tim Baumgärtner's work, found in [his repository](https://github.com/timbmg/Sentence-VAE) (link to the actual repository). Significant portions of the data processing code and some utility functions have been directly adopted from this source.

Architectural Modifications
---------------------------

Unlike the original LSTM-based model described by Bowman et al., this implementation utilizes * Transformer models * as the backbone for the encoder. This adjustment leverages the powerful contextual embeddings provided by Transformer architectures, enhancing the model's ability to capture complex syntactic and semantic patterns in text data.

Features
--------

*   **Latent Sentence Representation**: Maps sentences into a continuous latent space.
*   **Deterministic Decoding**: Converts latent representations back into text, ensuring the generation of well-formed sentences.
*   **Interpolation Capability**: Supports smooth interpolation between latent representations to explore new sentence constructions.

Getting Started
---------------

### Prerequisites

Ensure you have Python 3.8+ installed, along with the following packages:

*   torch
*   transformers
*   numpy
*   tqdm

### Installation

Clone the repository and install the required Python packages:

`git clone https://github.com/yourgithub/Sentence-VAE.git cd Sentence-VAE pip install -r requirements.txt`

### Usage

To train the model, adjust the settings in `config.json` or pass parameters directly through the command line:

`python main.py --mode train --epochs 50 --batch_size 32 --learning_rate 0.001`

For generating sentences from a trained model:

`python main.py --mode inference --load_checkpoint path/to/your/model.pth --num_samples 10`


### Results

Contributions
-------------

Contributions are welcome! If you find a bug or have suggestions for improvements, please open an issue or submit a pull request.
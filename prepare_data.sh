#!/bin/bash
# Script to download and prepare PTB dataset for the SentenceVAE-Transformers project

# Create data directory
mkdir -p data

# Download dataset
echo "Downloading the dataset..."
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz -O simple-examples.tgz

# Extract files
echo "Extracting files..."
tar -xf simple-examples.tgz

# Move necessary files
mv simple-examples/data/ptb.train.txt data/
mv simple-examples/data/ptb.valid.txt data/
mv simple-examples/data/ptb.test.txt data/

# Clean up
rm -rf simple-examples simple-examples.tgz

echo "Data preparation complete."

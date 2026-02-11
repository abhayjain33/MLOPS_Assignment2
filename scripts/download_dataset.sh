#!/bin/bash

# Script to download and prepare Cats vs Dogs dataset from Kaggle

set -e

echo "========================================="
echo "Kaggle Dataset Download Script"
echo "========================================="

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check for Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "⚠️  Kaggle API credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. This will download kaggle.json"
    echo "5. Move it to ~/.kaggle/kaggle.json"
    echo ""
    echo "Run these commands:"
    echo "  mkdir -p ~/.kaggle"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p data/raw_download
mkdir -p data/raw/cats
mkdir -p data/raw/dogs

# Download dataset
echo ""
echo "Downloading Cats vs Dogs dataset from Kaggle..."
echo "This may take a few minutes depending on your internet speed..."
cd data/raw_download

if [ ! -f "dogs-vs-cats.zip" ]; then
    kaggle competitions download -c dogs-vs-cats
else
    echo "Dataset already downloaded, skipping download..."
fi

# Extract dataset
echo ""
echo "Extracting dataset..."
if [ -f "dogs-vs-cats.zip" ]; then
    unzip -q dogs-vs-cats.zip
    
    # The competition zip contains train.zip and test1.zip
    if [ -f "train.zip" ]; then
        echo "Extracting training data..."
        unzip -q train.zip -d train_temp
        
        # Organize images into cats and dogs folders
        echo "Organizing images into cats and dogs folders..."
        
        # Move cat images
        echo "Processing cat images..."
        mv train_temp/cat.*.jpg ../raw/cats/ 2>/dev/null || true
        
        # Move dog images
        echo "Processing dog images..."
        mv train_temp/dog.*.jpg ../raw/dogs/ 2>/dev/null || true
        
        # Clean up
        rm -rf train_temp
        
        # Count images
        cat_count=$(ls ../raw/cats/*.jpg 2>/dev/null | wc -l)
        dog_count=$(ls ../raw/dogs/*.jpg 2>/dev/null | wc -l)
        
        echo ""
        echo "✓ Dataset prepared successfully!"
        echo "  Cats: $cat_count images"
        echo "  Dogs: $dog_count images"
        echo "  Total: $((cat_count + dog_count)) images"
        echo ""
        echo "Dataset location: data/raw/"
        
    else
        echo "Error: train.zip not found in the downloaded archive"
        exit 1
    fi
else
    echo "Error: dogs-vs-cats.zip not found"
    exit 1
fi

# Return to project root
cd ../..

echo ""
echo "========================================="
echo "✓ Download and setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Train the model: python src/train.py"
echo "2. View MLflow UI: mlflow ui"
echo ""

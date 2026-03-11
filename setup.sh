#!/bin/bash

echo "Setting up AutoML Research Prototype..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
echo "Installing dependencies..."
pip install -q scikit-learn pandas numpy scipy reportlab

# Create test dataset
echo "Creating test dataset..."
python3 << 'EOF'
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df.to_csv('wine_dataset.csv', index=False)
print(f"✓ Created wine_dataset.csv: {df.shape[0]} samples, {df.shape[1]} columns")
EOF

echo ""
echo "Setup complete! ✓"
echo ""
echo "Run the system with:"
echo "  ./run.sh wine_dataset.csv"
echo ""
echo "Or with custom options:"
echo "  ./run.sh wine_dataset.csv --output report.pdf --max-iterations 10"

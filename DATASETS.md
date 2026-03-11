# Recommended Test Datasets

## 1. Adult Income Dataset ✓ (Currently Tested)
- **Samples**: 32,561
- **Features**: 14 (6 numeric, 8 categorical)
- **Task**: Binary classification (income >50K or ≤50K)
- **Challenges**: 
  - Missing values (~1%)
  - High class imbalance (0.32)
  - Mixed feature types
  - Categorical encoding needed
- **Runtime**: ~8 seconds for 3 iterations
- **Best Accuracy**: 85.98%

**Why it's good**: Tests preprocessing, handles missing data, categorical encoding, imbalance

---

## 2. Bank Marketing Dataset (Recommended)
- **Samples**: 45,211
- **Features**: 16 (10 numeric, 6 categorical)
- **Task**: Binary classification
- **Challenges**: Severe class imbalance (11%), mixed types, larger scale

```bash
# Download
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv
mv bank-additional-full.csv bank_marketing.csv
```

---

## 3. Credit Card Default Dataset
- **Samples**: 30,000
- **Features**: 23 (numeric + categorical)
- **Task**: Binary classification
- **Challenges**: Imbalanced, financial data, multiple correlated features

```bash
# Download from UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
# Convert to CSV manually or use pandas
```

---

## 4. California Housing (Regression)
- **Samples**: 20,640
- **Features**: 8 (all numeric)
- **Task**: Regression
- **Challenges**: Tests regression path, larger dataset, feature scaling important

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target
df.to_csv('california_housing.csv', index=False)
```

---

## 5. Covertype Dataset (Multi-class)
- **Samples**: 581,012
- **Features**: 54
- **Task**: Multi-class classification (7 classes)
- **Challenges**: Large scale, high dimensionality, tests performance

---

## Quick Test Script

Want to test multiple datasets? Create this:

```bash
#!/bin/bash
# test_all.sh

echo "Testing multiple datasets..."

./run.sh adult_income.csv --output reports/adult_report.pdf
./run.sh california_housing.csv --output reports/housing_report.pdf
./run.sh bank_marketing.csv --output reports/bank_report.pdf

echo "All tests complete!"
```

---

## Current Test Results

### Wine Dataset (Too Simple)
- 178 samples, 13 features
- Perfect accuracy (1.0) immediately
- No missing values, no categorical features
- Runtime: 0.2s
- **Problem**: Too easy, doesn't showcase system

### Adult Income Dataset (Good)
- 32,561 samples, 14 features
- Realistic accuracy (85.98%)
- Missing values, categorical features, imbalance
- Runtime: 8s
- **Perfect for demo**: Shows all system capabilities

---

## Recommendation

**Use Adult Income for your demo** - it's the sweet spot:
- Large enough to show scalability
- Complex enough to show preprocessing
- Has missing values (tests imputation)
- Has categorical features (tests encoding)
- Has class imbalance (shows data quality analysis)
- Realistic accuracy (not perfect, shows model differences)
- Fast enough for live demo (~8 seconds)

The meta-features will also be more interesting with this dataset.

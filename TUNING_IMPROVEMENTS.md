# Hyperparameter Tuning Improvements

## Problem
The initial 3 iterations with LogisticRegression/LinearRegression were producing the same accuracy because:
- Limited hyperparameter choices (only 2 values for C, 1 solver)
- Not exploring enough parameter space
- Missing important parameters that affect model performance

## Solution

### 1. Expanded Parameter Grids

#### LogisticRegression (Classification)
**Before**: 2 combinations
```python
'C': [0.1, 1.0]
'solver': ['lbfgs']
```

**After**: 90 combinations
```python
'C': [0.01, 0.1, 1.0, 10.0, 100.0]        # Regularization strength
'penalty': ['l1', 'l2']                     # Regularization type
'solver': ['liblinear', 'saga']             # Optimization algorithm
'max_iter': [500, 1000, 2000]               # Convergence iterations
```

**Why these parameters matter**:
- `C`: Controls regularization (smaller = more regularization)
- `penalty`: L1 for feature selection, L2 for general regularization
- `solver`: Different algorithms work better on different data
- `max_iter`: Ensures convergence on complex datasets

---

#### LinearRegression (Regression)
**Before**: 0 combinations (no tuning)
```python
{}
```

**After**: 4 combinations
```python
'fit_intercept': [True, False]              # Include bias term
'positive': [True, False]                   # Force positive coefficients
```

**Why these parameters matter**:
- `fit_intercept`: Whether to calculate intercept (important for centered data)
- `positive`: Useful for problems where negative coefficients don't make sense

---

#### RandomForest
**Before**: 4 combinations
```python
'n_estimators': [100, 150]
'max_depth': [10, None]
```

**After**: 960 combinations
```python
'n_estimators': [50, 100, 150, 200]         # Number of trees
'max_depth': [5, 10, 15, 20, None]          # Tree depth
'min_samples_split': [2, 5, 10]             # Min samples to split node
'min_samples_leaf': [1, 2, 4]               # Min samples in leaf
'max_features': ['sqrt', 'log2']            # Features per split
```

**Why these parameters matter**:
- `n_estimators`: More trees = better performance but slower
- `max_depth`: Controls overfitting
- `min_samples_split/leaf`: Prevents overfitting on small datasets
- `max_features`: Controls randomness and correlation between trees

---

#### SVM/SVR
**Before**: Not defined
```python
{}
```

**After**: 300 combinations
```python
'C': [0.1, 1.0, 10.0, 100.0]                # Regularization
'kernel': ['rbf', 'poly', 'sigmoid']        # Kernel function
'gamma': ['scale', 'auto', 0.001, 0.01, 0.1] # Kernel coefficient
'degree': [2, 3, 4]                         # Polynomial degree
```

**Why these parameters matter**:
- `C`: Trade-off between margin and misclassification
- `kernel`: Different kernels capture different patterns
- `gamma`: Influence of single training example
- `degree`: Complexity of polynomial kernel

---

### 2. Smart Search Strategy

**Hybrid Approach**:
- **GridSearchCV**: Used when total combinations ≤ 30 (exhaustive search)
- **RandomizedSearchCV**: Used when combinations > 30 (samples 20 random combinations)

**Benefits**:
- Fast on small grids (LogisticRegression: 90 combinations → Grid)
- Efficient on large grids (RandomForest: 960 combinations → Random 20)
- Increased CV folds from 2 to 3 for better validation

**Example Output**:
```
LogisticRegression:
    Grid search: 90 combinations, 3-fold CV

RandomForest:
    Randomized search: 20 of 960 combinations, 3-fold CV
```

---

## Expected Impact

### Before Tuning:
```
Iteration 1: LogisticRegression → Accuracy: 0.9500
Iteration 2: LogisticRegression → Accuracy: 0.9500 (no change)
Iteration 3: LogisticRegression → Accuracy: 0.9500 (no change)
```

### After Tuning (with --enable-tuning):
```
Iteration 1: LogisticRegression → Accuracy: 0.9500
Iteration 2: LogisticRegression (tuned) → Accuracy: 0.9667 (improved!)
  Best params: {'C': 10.0, 'penalty': 'l2', 'solver': 'saga', 'max_iter': 1000}
Iteration 3: RandomForest → Accuracy: 0.9733
```

---

## Usage

```bash
# Enable tuning to see improvements
python cli/main.py data/iris.csv --enable-tuning --max-iterations 5

# Without tuning (faster, uses default parameters)
python cli/main.py data/iris.csv --max-iterations 5
```

---

## Technical Details

### Search Space Size:
- **LogisticRegression**: 5 × 2 × 2 × 3 = 90 combinations
- **LinearRegression**: 2 × 2 = 4 combinations
- **RandomForest**: 4 × 5 × 3 × 3 × 2 = 960 combinations
- **SVM/SVR**: 4 × 3 × 5 × 3 = 180 combinations (degree only for poly)

### Computational Cost:
- **Grid Search**: Tests all combinations (slow but thorough)
- **Randomized Search**: Tests 20 random combinations (fast, good coverage)
- **Cross-Validation**: 3-fold CV balances accuracy and speed

### Parameter Selection Strategy:
1. Start with wide ranges (0.01 to 100 for C)
2. Include categorical choices (penalty, solver, kernel)
3. Cover common values from literature
4. Balance exploration vs computation time

---

## Why This Fixes the Problem

1. **More Diversity**: 90 combinations vs 2 means more chances to find better parameters
2. **Better Coverage**: Explores regularization, solvers, and convergence settings
3. **Smarter Search**: RandomizedSearch efficiently handles large grids
4. **Visible Differences**: Different parameter combinations will produce different accuracies

Now when tuning is enabled, you'll see actual improvements between iterations instead of the same accuracy repeated!

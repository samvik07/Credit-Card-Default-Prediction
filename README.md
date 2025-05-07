# Credit Card Default Prediction Model

A machine learning project to predict credit card defaulters using various models and evaluation strategies.
Focuses on domain-specific metrics like Recall (Class 1) and Precision (Class 0), beyond just accuracy.

## Project Structure

- `eda/`: Exploratory data analysis notebook
- `data/`: Data loading utilities
- `preprocessing/`: Feature encoding, SMOTE, and other transformations
- `modeling/`: Model training and tuning scripts
- `evaluation/`: Evaluation and metric plotting

## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter tuning using:
  - Accuracy (default)
  - Domain-specific custom score

## Evaluation Focus

- Recall of Class 1 (catching defaulters)
- Precision of Class 0 (preserving creditworthy leads)

## Key Learnings

- Accuracy can be misleading in imbalanced datasets.
- Business context determines the right evaluation metric.

## How to Run

```bash
# Create environment (optional)
pip install -r requirements.txt

# Run full pipeline
python main.py
```

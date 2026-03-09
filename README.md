# CloudWatch Intrusion Detector (Assignment Starter)

Starter project for a network-traffic intrusion detection assignment using tabular AWS CloudWatch-style logs.

## Dataset options
- https://www.kaggle.com/datasets/startingsecurity/cybersecurity-honeypot-attacks
- https://www.kaggle.com/datasets/jancsg/cybersecurity-suspicious-web-threat-interactions

Place your CSV file in `data/` (for example `data/your_dataset.csv`).

## Project layout
- `notebooks/intrusion_detector.ipynb`: step-by-step notebook for Parts 1-6
- `src/train.py`: end-to-end training/evaluation pipeline
- `outputs/`: generated metrics, plots, and failure-case files

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (script mode)
```bash
python src/train.py --data data/your_dataset.csv --target label
```

Notes:
- If your label column is not named `label`, pass `--target <column_name>`.
- If categorical target labels need explicit mapping, pass:
  `--positive-labels attack,malicious,intrusion`
- If one-hot encoding becomes too large, lower/raise `--max-cat-cardinality`.

## Outputs produced
- `outputs/training_curves.png`: training/validation loss + accuracy
- `outputs/model_comparison.csv`: NN vs baseline metrics
- `outputs/metrics_nn.json`: final NN test metrics
- `outputs/metrics_baseline.json`: baseline test metrics
- `outputs/failure_cases.csv`: up to 5 misclassified examples with failure tags
- `outputs/run_metadata.json`: split details and preprocessing metadata

## How this maps to assignment parts
1. **Define the problem**: write answers in notebook markdown cells (Part 1).
2. **Data prep + features**: handled in notebook and `src/train.py` (imputation, encoding, scaling, split).
3. **Neural network**: PyTorch feedforward NN with <=3 hidden layers, ReLU, sigmoid output.
4. **Baseline comparison**: Logistic Regression baseline + metric comparison table.
5. **Failure case analysis**: exports misclassified examples and starter failure categories.
6. **Analysis**: final markdown response section in notebook.


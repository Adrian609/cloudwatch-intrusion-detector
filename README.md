# CloudWatch Intrusion Detector (Assignment Starter)

Starter project for a network-traffic intrusion detection using tabular AWS CloudWatch-style logs.

## Dataset
Cybersecurity: Suspicious Web Threat Interactions
- https://www.kaggle.com/datasets/jancsg/cybersecurity-suspicious-web-threat-interactions

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



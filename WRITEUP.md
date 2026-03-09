# Intrusion Detection Analysis

## Part 1 - Define the Problem

### What attacker behavior are we trying to detect?
The goal is to detect malicious or suspicious behavior in CloudWatch-style network telemetry, such as scanning/probing patterns, repeated connection attempts to sensitive service ports, and infrastructure interactions associated with known attack workflows.

### What does a false positive cost?
False positives increase analyst workload and alert fatigue, consume incident response time, and can lead to unnecessary blocking of legitimate traffic.

### What does a false negative cost?
False negatives are more severe: missed intrusions can allow persistence, credential abuse, lateral movement, data exfiltration, and downstream compliance/reputation damage.

### Why might a neural network help (or not help)?
A neural network can learn nonlinear relationships among engineered traffic features. However, for tabular security logs, simpler models (for example logistic regression or tree-based models) are often competitive and easier to interpret. In this project, the baseline performed as well as the NN.

---

## Part 2 - Data Preparation & Feature Engineering

Dataset used: `data/CloudWatch_Traffic_Web_Attack.csv`.

Steps completed:
1. Loaded data and normalized column names.
2. Removed duplicate rows.
3. Built binary target from `protocol`:
   - Positive class (`1`): `mssqld,httpd,mysqld,pptpd,ftpd,mongod,epmapper,mqttd`
   - Negative class (`0`): remaining protocol (`smbd`)
4. Dropped constant columns (`transport`, `type`).
5. Split into train/validation/test using stratified splits (~70/15/15).
6. Numeric pipeline: median imputation + standardization.
7. Categorical pipeline: most-frequent imputation + one-hot encoding.
8. High-cardinality categorical columns dropped (`src_port`, `timestamp`) to control feature explosion.

Run metadata:
- Rows original: 27,574
- Rows after dedup: 27,574
- Train rows: 19,300
- Validation rows: 4,137
- Test rows: 4,137

---

## Part 3 - Neural Network Classifier

Model type: fully connected feedforward network (PyTorch), meeting assignment constraints.

Architecture:
1. Input layer (tabular encoded features)
2. Hidden layer 1: 128 units, ReLU
3. Hidden layer 2: 64 units, ReLU
4. Hidden layer 3: 32 units, ReLU
5. Output layer: 1 unit, Sigmoid

Training setup:
- Loss: binary cross-entropy
- Optimizer: Adam
- Class weighting: balanced
- Early stopping on validation loss

Final NN test performance:
- Accuracy: 1.000
- Precision: 1.000
- Recall: 1.000
- F1: 1.000
- False positives: 0
- False negatives: 0

Training curves are saved in `outputs/training_curves.png`.

---

## Part 4 - Baseline Comparison

Baseline model: Logistic Regression (class-weight balanced).

### Metrics comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Neural Network | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |

Interpretation:
- Both models achieved identical scores on this dataset split.
- Given equal performance, the simpler baseline is preferable for interpretability and operational simplicity.

---

## Part 5 - Failure Case Analysis

At the default decision threshold (0.5), the NN produced **0 misclassifications** on the test split (`outputs/failure_cases.csv`).

Because the assignment asks for at least 5 misclassified examples *if possible*, I performed a threshold stress test (threshold = 0.9999) and obtained 5 false negatives (`outputs/failure_cases_threshold_0_9999.csv`).

### 5 misclassified examples (stress-test threshold)

| # | dst_port | src_ip | timestamp | true | pred | p(malicious) | Failure type |
|---|---:|---|---|---:|---:|---:|---|
| 1 | 21 | 192.241.221.109 | 2022-05-07T17:52:02.401439 | 1 | 0 | 0.999709 | Ambiguity / threshold calibration |
| 2 | 1723 | 192.241.212.223 | 2022-05-07T17:51:36.561668 | 1 | 0 | 0.999757 | Ambiguity / threshold calibration |
| 3 | 135 | 198.235.24.10 | 2022-05-07T20:09:03.874985 | 1 | 0 | 0.999744 | Ambiguity / threshold calibration |
| 4 | 3306 | 185.162.235.162 | 2022-05-07T16:33:05.348748 | 1 | 0 | 0.999848 | Ambiguity / threshold calibration |
| 5 | 1433 | 5.190.78.249 | 2022-05-07T16:12:39.725731 | 1 | 0 | 0.999618 | Ambiguity / threshold calibration |

How an attacker could exploit these weaknesses:
1. Operate close to policy boundaries to exploit poor threshold calibration.
2. Mimic traffic patterns that resemble high-volume benign activity.
3. Use protocol/port profiles that are common enough to avoid strong anomaly signals.

Note on dataset limitations:
- The chosen data required engineered labels from protocol families, and results are likely optimistic.
- Real-world evaluation should use independently labeled benign + malicious traffic from different time windows.

---

## Part 6 - Analysis
### Should this model operate fully autonomously?
Not for high-impact actions. It can prioritize and score alerts autonomously, but blocking/quarantine decisions should remain human-reviewed or policy-gated.

### How should analysts interact with the system?
Analysts should receive ranked alerts with confidence scores, key contributing features, and quick links to supporting log evidence. Their feedback should be captured for periodic retraining and threshold tuning.

### What are the ethical risks of false accusations?
False accusations can block legitimate users, disrupt operations, and unfairly target individuals or organizations. They can also damage trust in security tooling and incident response teams.

### How could model bias manifest in cybersecurity?
Bias can appear through skewed training data and proxy features (for example geography, ASN, hosting provider, or device type), causing systematic over-flagging of certain populations or environments.

---

## Files Generated

- `outputs/metrics_nn.json`
- `outputs/metrics_baseline.json`
- `outputs/model_comparison.csv`
- `outputs/training_curves.png`
- `outputs/failure_cases.csv`
- `outputs/failure_cases_threshold_0_9999.csv`
- `outputs/run_metadata.json`

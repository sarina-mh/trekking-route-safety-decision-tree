# Trekking Route Safety Decision Tree

Binary classification of mountain trekking routes as safe or dangerous using a constrained decision tree, trained on environmental and hiker-condition features. Evaluates whether a composite risk feature (`environmental_danger`) improves predictive performance over a baseline model.

## Overview

This project applies scikit-learn's `DecisionTreeClassifier` to a synthetic dataset of 6,000 trekking expeditions. The goal is to predict route safety (`is_safe`: 0 = dangerous, 1 = safe) from four input features: terrain slope, rainfall, wolf-encounter probability, and hiker cold-resistance level.

A key contribution is the engineered feature `environmental_danger`, which captures the multiplicative interaction between slope steepness, rainfall intensity, and wildlife hazard, reasoning that these risk factors compound rather than act independently.

## Mathematical Framework

The composite hazard term is defined as:

```
environmental_danger = slope_angle × rain_mm × (1 + wolf_prob)
```

- `slope_angle` (degrees): terrain steepness, higher values increase fall risk.
- `rain_mm` (mm): rainfall accumulation, higher values reduce visibility and traction.
- `wolf_prob` (0 to 1): probability of wolf encounter on the route.

The `(1 + wolf_prob)` multiplier ensures the term remains positive even when wolf probability is zero, while scaling linearly with wildlife risk. The multiplicative structure reflects the hypothesis that danger factors are synergistic: a steep slope in heavy rain with high wolf probability is far more hazardous than the sum of individual risks.

## Dataset

`data/trekking_expedition.csv`: 6,000 synthetic records with 5 columns:

| Column | Type | Description |
|---|---|---|
| `slope_angle` | float | Terrain slope in degrees (≥ 0) |
| `wolf_prob` | float | Wolf encounter probability [0, 1] |
| `rain_mm` | float | Rainfall in millimeters (≥ 0) |
| `cold_resistance` | categorical | Hiker cold resistance: Low, Medium, or High |
| `is_safe` | binary | Route safety label: 0 (dangerous) or 1 (safe) |

## Installation

```bash
# Clone the repository
git clone https://github.com/Sarina/trekking-route-safety-decision-tree.git
cd trekking-route-safety-decision-tree

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

```bash
# Launch the notebook
jupyter notebook notebooks/trekking_route_safety_decision_tree.ipynb
```

Run all cells to reproduce the full pipeline: data cleaning, feature engineering, model training, evaluation, and decision tree visualization.

## Results

Two experiments are compared:

| Model | Accuracy | F1-score |
|---|---|---|
| Baseline (4 features) | TBD | TBD |
| With `environmental_danger` (5 features) | TBD | TBD |

*Run the notebook to populate the table with current results.*

The decision tree is constrained to `max_depth=4` with `min_samples_split=25` and `min_samples_leaf=12` to produce interpretable split rules. The fitted tree is visualized at the end of the notebook.

### Decision Tree Visualization

![Decision Tree with environmental_danger feature](images/decision_tree.png)

## Project Structure

```
trekking-route-safety-decision-tree/
├── data/
│   └── trekking_expedition.csv
├── images/
│   └── decision_tree.png
├── notebooks/
│   └── trekking_route_safety_decision_tree.ipynb
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Citation

```bibtex
@software{trekking_route_safety_2026,
  author       = {Sarina},
  title        = {Trekking Route Safety Decision Tree},
  year         = {2026},
  url          = {https://github.com/Sarina/trekking-route-safety-decision-tree},
  license      = {MIT}
}
```

## License

[MIT](LICENSE)

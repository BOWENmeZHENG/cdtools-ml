# CDTools-ML

Adding ML functionality to CDTools for enhanced ptychography reconstruction.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Install the package in development mode:
```bash
pip install -e .
```

## Examples

### Gold Ball Ptychography

Run the example without ML:
```bash
python examples/gold_ball_ptycho_ml.py
```

Run the example with ML:
```bash
python examples/gold_ball_ptycho_ml.py --USE_ML
```

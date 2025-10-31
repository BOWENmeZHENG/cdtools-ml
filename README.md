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

### Using multiple incoherently mixing probe modes

Run the example without ML:
```bash
python python run_fancy.py
```

Run the example with ML:
```bash
python python run_fancy.py --USE_ML
```

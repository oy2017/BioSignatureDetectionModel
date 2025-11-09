# BioSignatureDetectionModel
Train and evaluate ML models to detect bio signatures from spectra of exoplanet that is similar as the ones that will be studied in Ariel Space Mission in 2029. 

## Getting Started

Follow these steps to set up the project and run the analysis scripts.

### 1. Set up the Python Environment

Create a virtual environment and install the required dependencies (including MultiREx):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Data Generation and Evaluation Scripts

Once the environment is set up, you can run the data generation and evaluation scripts.

#### Generate Multiverse Data

Use the `generate_multiverse_data.py` script to generate separate training and testing datasets. Specify the fill gas (e.g., `H2` or `N2`) and the purpose (`train` or `test`).

To generate the **training** dataset (3000 universes) for H2:
```bash
source venv/bin/activate
python generate_multiverse_data.py H2 --purpose train
# This will create 'multirex_spectra_H2_train.parquet'
```

To generate the **testing** dataset (600 universes) for H2:
```bash
source venv/bin/activate
python generate_multiverse_data.py H2 --purpose test
# This will create 'multirex_spectra_H2_test.parquet'
```
Repeat the process for `N2` or any other fill gas.

#### Evaluate Data

Use the `evaluate_random_forest.py` script to train and evaluate the model. The script will automatically load the corresponding `_train.parquet` and `_test.parquet` files.

To evaluate the H2 dataset:
```bash
source venv/bin/activate
python evaluate_random_forest.py H2
```

To evaluate the N2 dataset:
```bash
source venv/bin/activate
python evaluate_random_forest.py N2
```

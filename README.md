# BioSignatureDetectionModel
Train and evaluate ML models to detect bio signatures from spectra of exoplanet that is similar as the ones that will be studied in Ariel Space Mission in 2029.. 

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

#### H2 Data Generation and Evaluation

To generate the H2 dataset and evaluate the model:

```bash
source venv/bin/activate
python generate_h2_data.py
python evaluate_h2_data.py
```

#### N2 Data Generation and Evaluation

To generate the N2 dataset and evaluate the model:

```bash
source venv/bin/activate
python generate_n2_data.py
python evaluate_n2_data.py
```
# DMH-JNMF: Deep Multi-Head Self-Attention Encoder with Joint NMF for Alzheimer's Disease
📖 Overview
The DMH-JNMF framework is designed to overcome critical computational bottlenecks in brain imaging genetics. By synergizing a Deep Multi-Head Self-Attention Encoder (implemented in PyTorch) with an HSIC-constrained Joint NMF (implemented in MATLAB), this model explicitly decouples spatially heterogeneous pathological features and captures high-order multi-omics dependencies for the early diagnosis of Alzheimer's Disease (AD).

🗂️ Repository Structure
Based on our multi-stage pipeline, the codebase is divided into two main components: deep feature extraction (Python) and matrix factorization (MATLAB).

Plaintext
├── DSAE/                          # Part 1: Deep Self-Attention Encoder
│   ├── data/                      # Directory for raw multi-modal inputs (MRI.csv, FDG.csv, etc.)
│   ├── reconstructed/             # Directory for outputting latent representations
│   └── Autoencoder.py             # PyTorch implementation of the deep encoder
│
└── MACJNMF-code/                  # Part 2: Matrix Factorization & Module Extraction
    ├── main.m                     # Main execution script for the factorization process
    ├── MACJNMF.m                  # Core Joint NMF algorithm implementation
    ├── BrainNet_Node/             # Assets for macroscopic network visualization
    └── [reconstructed_data].csv   # Inputs generated from the Python module

💻 Requirements & Environment
To ensure reproducibility, please configure your environment as follows:

For Part 1 (Deep Learning Encoder):

Python >= 3.8
PyTorch (version 2.1 recommended)
Hardware: NVIDIA GPU (Tested on GeForce RTX 4050 / RTX 3090)

For Part 2 (Matrix Factorization):

MATLAB (version R2024a recommended)

🚀 Quick Start
Step 1: Deep Feature Extraction (Python)
Navigate to the DSAE directory.
Ensure your raw modality data (MRI.csv, FDG.csv, AV45.csv, Gene.csv) are placed inside the DSAE/data/ folder.
Run the autoencoder script:

Bash
cd DSAE
python Autoencoder.py
The trained model will generate non-linear latent representations (e.g., full_MRI_recon1.csv) and save them automatically to the DSAE/reconstructed/ folder.

Step 2: Joint NMF & Module Extraction (MATLAB)
Copy the reconstructed .csv files from DSAE/reconstructed/ into the MACJNMF-code/ directory.
Open MATLAB and set MACJNMF-code/ as your current working directory.
Run the main script:
Matlab
main
The script will execute the MACJNMF.m function to perform matrix factorization and output the core coefficient matrices (e.g., H_Gene.csv) for downstream biological pathway analysis and BrainNet visualization.

📊 Data Availability
The multi-modal dataset (sMRI, FDG-PET, AV45-PET, and Gene expression profiles) used to validate this framework was obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. To access the data, researchers must apply directly through the ADNI website(https://adni.loni.usc.edu/).

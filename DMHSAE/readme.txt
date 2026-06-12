This repository implements a deep learning pipeline for reconstructing latent features from four biomedical modalities (MRI, FDG-PET, AV45-PET, and Gene expression) using an attention-guided autoencoder network, followed by saving cross-validated, fused latent representations for downstream multimodal modeling (e.g., joint NMF).

1.Project Structure
├── Autoencoder.py                 # Deep Encoder-Decoder model with Residual & Attention modules
├── Train.py         # Main script for training & reconstruction across 5 folds
├── utils.py                      # Utilities: Dataloader, EarlyStopping, Normalization, Metrics
├── reconstructed/                # Output directory for saving latent features per fold
├── MRi.csv / FDG.csv / AV45.csv / Gene.csv   # Input multimodal data (196 x D matrices)
├── full_MRI_recon.csv/full_AV45_recon.csv/full_FDG_recon.csv/full_Gene_recon.csv

2.Dependencies
Ensure you have the following packages:
pip install torch numpy pandas scikit-learn scipy

3.Getting Started
（1）Place the following CSV files in your project root:
MRi.csv
FDG.csv
AV45.csv
Gene.csv
（2）Train and Reconstruct：
python main_reconstructor.py
（3）Merge Reconstructed Features
./reconstructed/full_MRI_recon.csv
./reconstructed/full_FDG_recon.csv
./reconstructed/full_AV45_recon.csv
./reconstructed/full_Gene_recon.csv

#DASE-MACJNMF
This project proposed a deep self-attention enhanced multi-attribute constrained joint non-negative matrix factorization (DSAE-MACJJNMF) model  to explore hidden imaging genetic association in Alzheimer's disease. 
The code in this project will reproduce the results in our paper, "*Deep Self-Attention Enhanced Multi-Attribute Constrained Joint Non-Negative Matrix Factorization for Exploring Hidden Imaging Genetic Association in Alzheimer's Disease*". The code was implemented and tested using Matlab
You can run it by the .zip archive named DSAE-MACJNMF. The *DSAE-MACJNMF.zip* contains the *DSAE* file and *MACJNMF* file. 
1. You can run the DSAE model using *Train.m* in the DSAE file.The input files include *MRI.csv*, *FDG.csv*, *AV45.csv* and *Gene.csv*.The output is the reconstructed four files. Afterwards, the reconstructed files are used as inputs to the MACJNNMF model.

2. The reconstructed datasets were normalized to obtain *full_AV45_recon1.csv*, *full_MRI_recon1.csv*,  *full_FDG_recon1.csv* and *full_Gene_recon1.csv* in the DSAE-reconstructed file. After that the MACJNMF model can be implemented by running the *main.m* file in the MACJNMF file and the output is the decomposed *W* and three *H* matrices.

3. Installation
No installation steps are required beyond having the appropriate version of MATLAB and necessary dependencies.

However, the following libraries and toolboxes are recommended:

MATLAB (R2017a or later)

Optimization Toolbox (for advanced matrix optimization)

Statistics and Machine Learning Toolbox (for data processing)

Simply download the MATLAB scripts and place them in a directory for easy access.

4. Customizing Parameters:
K: The number of latent factors (default is 8, adjust based on your data).
lambda1, lambda2, lambda3: Regularization parameters for the different terms in the objective function.
alpha, beta, beta_param: Parameters controlling label alignment and modular constraints.
tt: Z-score thresholds for feature selection (one for each modality).
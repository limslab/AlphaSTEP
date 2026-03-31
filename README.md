# AlphaSTEP

### AlphaSTEP is a set of data prediction tools for predicting potential stereoisomers as well as RT and CCS values from peptide data, which is developed by LimsLab @ Nankai University (The Gongyu Li Research Group), through collaboration with Dr. Haohao Fu @ Nankai University. 

### Code Availability. All code used in this study is available on GitHub (https://github.com/limslab). AlphaSTEP is released under the GNU General Public License v3.0 (GPL-3.0). The repository contains Python scripts for data processing and analysis. The code was developed using Python 3.12, with key dependencies listed in the requirements.txt file. Users may install the required dependencies via pip (pip install -r requirements.txt) or conda (conda install --file requirements.txt). A comprehensive README.md provides step-by-step instructions for reproducing our analyses. The raw data processed by this code is available from both the Supporting Information and the GitHub repository. For any questions regarding the code, please contact Prof. Dr. Gongyu Li (ligongyu@nankai.edu.cn). 

AlphaSTEP prediction workflow:
1. Data Acquisition: Obtain raw protein sequence files (.fasta) from UniProt, NCBI, or other sources for simulated protease digestion and theoretical peptide generation (demo data: uniprotkb_human_disease_protein.fasta).
2. Data Processing:
a. Execute the "fasta" and "import" scripts to perform virtual digestion. Choose the protease (default: trypsin) and set digestion parameters. The scripts produce a file of theoretical precursor peptides with sequence, charge, protein/gene name, etc. (demo data: precursor_df_output.xlsx).
b. Further process the xlsx output to curate a non-redundant peptide list: Execute the "duplicate_removal" script to remove duplicate peptide entries. Execute the "remove_inclusion_relationship" script to identify peptides with sequence-inclusion relationships and retain only the shorter peptides. The resulting curated unique-peptide list is used as the input for isomer prediction (demo data: unique_peptide.xlsx).
3. Model Training:
a. Execute the "000_gen_train_set" script to combine L-type and D-type peptide records and assign class labels.
b. Execute the "001_data_process" script to read peptide sequences and labels, perform tokenization/encoding, split datasets, and apply class-balancing strategies.
c. Execute the "002_train" script to build and train the classification model on balanced data. Typical training features include class weights and learning-rate scheduling. Choose the best model by validation metrics.
d. Execute the "003_explain" script to load the trained model and perform attribution and interpretability analysis.
e. Execute the "004_visualize" script to visualize attribution results and highlight sequence motifs or key positions driving predictions (demo data: advanced best_model.pth).
4. Isomer Prediction:
a. Execute the "prediction" script to score candidate peptides in unique_peptide.xlsx: Convert relevant rows from the .xlsx file to a .txt input (one sequence per line) or otherwise transform into the model input format. Encode sequences numerically, apply the best model to extract sequence patterns and predict isomerization probability for each peptide. 
b. Export batch results and retain peptides with predicted isomer probability > 0.5 as candidate isomeric peptides (demo data: Isomeric_peptides.xlsx).
5. Retention Time (RT) and Collision Cross Section (CCS) Prediction:
a. Execute the "all_in_one_final" script to perform RT/CCS modeling and prediction. Predictions for D- and L-type peptides are performed independently for RT and CCS, resulting in four separate prediction tasks. The input Excel file consists of two sheets: the first sheet provides data for model training, and the second sheet lists the peptide sequences to be predicted (demo data: train_set.xlsx).
b. Workflow inside the notebook: 
Unify sequence and numerical feature preprocessing for L- and D-type peptides; Split training/validation sets, perform standardization/normalization; Convert numerical features to binary/numpy files for efficient loading; Build a multimodal ensemble regression model to predict CCS and RT values, this model captures peptide physicochemical features and sequence-derived embeddings; Train and validate the regression model, then run batched prediction on candidate peptides (demo data: L-CCS.ckpt). Export the predicted RT/CCS results (demo data: predict_results.xlsx).
c. Final merge: Combine prediction scores and predicted RT/CCS into a single comprehensive file containing peptide sequence, protein name, isomer probability, predicted CCS, predicted RT, and additional metadata (demo data: result.xlsx).


Done!




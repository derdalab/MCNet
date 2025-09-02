# MCNet

This repository accompanies the paper Atom-level machine learning of
protein-glycan interactions and cross-chiral recognition in glycobiology.

Eric J. Carpenter, Chuanhao Peng, Simatsidk Haregu, Logan Woudstra, Amika Sood,
Nicholas Twells, Robert J. Woods, Lara K. Mahal, Sheng-Kai Wang,
Russell Greiner, Ratmir Derda

A preprint is available at https://www.biorxiv.org/content/10.1101/2025.01.21.633632

In addition to scripts used to train ML models, this repository includes 
model parameters of pre-trained versions of the models
scripts used to process the source data prior to input,
scripts extract data from the output files
and scripts to use the pre-trained models for inference on new molecules.


Related work is available in repositories:

https://github.com/loganwoudstra/mirror-image-glycans
https://github.com/loganwoudstra/PEER_Benchmark


Python scripts have been developed and run with:

1. Python: 3.7.3, 3.8.10, 3.12.3	https://www.python.org
2. PyTorch: 1.8.0, 1.9.0, 2.3,0	https://pytorch.org
3. RDKit: 2018.09, 2019.09.1, 2023.09.6	https://www.rdkit.org
4. XGBoost: 3.0.2		https://xgboost.ai
5. NumPy: 1.17.45, 1.26.4		https://numpy.org
6. SciPy: 1.3.3, 1.7.1, 1.13.1	https://scipy.org
7. Pandas:	0.25.3, 2.2.2	https://pandas.pydata.org

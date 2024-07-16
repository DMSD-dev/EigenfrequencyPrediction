# EigenfrequencyPrediction
 predict realistic eigenfrequencies of railway bridges
 
This is an implementation of the paper "A Machine Learning Based Algorithm for the Prediction of Eigenfrequencies of Railway Bridges" using Python 3 and Pytorch. The model predicts a more realistic first bending eigenfrequency (natural frequency) of single-span, single-track filler beam bridges using polynomial regression, ANN and XGBoost.

The repository contains:
- Source code of the model created with Pytorch 
- Training code for the simulations
- xxx

The code is documented and designed to make the paper more comprehensible and easily extendable. If these repros are used in your research, please consider citing this repository.
# Installation
- Clone this repository
- Install dependencies

`pip install -r requirements.txt`

- Run setup from the repository root directory

`python3 setup.py install`

# Code Overview
The code is organised as follows:
- **data:** Contains the training data used for the simulation (*dataset_0_30_100_200_1.npz*), as well as for the Schmutter. Furthermore, a smaller dataset (*debug_dataset.npz*) is included for simple debugging.
- **experiments:** Contains the experiments used:
- **models:** Contains the models used.
- **tmb:** Contains the code of the package.

# Citation
Use this bibtex to cite this repository:  
@misc{EigenfrequencyPrediction_2024,  
  title={Eigenfrequency Prediction},  
  author={G. Grunert, D. Grunert, R. Behnke, S. Schäfer, X. Liu, S.R. Challagonda},  
  year={2024},  
  publisher={Github},  
  journal={GitHub repository},  
  howpublished={\url{https://github.com/DMSD-dev/EigenfrequencyPrediction }},  
}  

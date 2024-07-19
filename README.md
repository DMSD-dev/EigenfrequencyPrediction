# EigenfrequencyPrediction
 predict realistic eigenfrequencies of railway bridges
 
This is an implementation of the findings of the paper "A Machine Learning Based Algorithm for the Prediction of Eigenfrequencies of Railway Bridges" using Python 3. The model predicts a more realistic first bending eigenfrequency (natural frequency) of single-span, single-track filler beam bridges using XGBoost.

The repository contains:
- Files to and python files enabling the application of the final XGBoost model obtained as a result of the aforementioned paper to predict the eigenfrequencies of railway bridges 

**Note**: Please note the discussed application limitations in the paper. Currently, we cannot further facilitate preprocessing and filtering as described, because this would require the publication of non-public data as reference material which is unfortunately not possible. We are still working on finding an alternative solution.

The code is documented and designed to enable the applications of the findings of the paper to novel datasets even by non-technical parties. 
# Installation
- Clone or download this repository
- Install dependencies using

`pip install -r requirements.txt`

Note that this requires an installation of the python programming language as well as the pip-package manager.

# Code Overview
The code is organised as follows:
- **main.py** enables the application of the final model to predict eigenfrequencies based on command-line arguments
- **XGBoostPrediction.py** further facilitates ways to apply the model in ways used in the paper methodology
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

# An improved Forest change detection in Sentinel-2 satellite images using Attention Residual U-Net
This is a repositiory for the codes generated from our experiment on An improved Forest change detection in Sentinel-2 satellite images using Attention Residual U-Net  

# Contents of the Repository
Other than the readme file, the repository contains 2 files named: models.py and AttResUnet_training.py

# The Model file (model.py)
This file contains the models and loss functions. all the three models have been defined: Standard U-Net, Attention U-Net and the Attention Residual U-Net.

# The Attention Residual U-Net file (AttResUnet_training.py) 
This file contains the codes to run the three models mentioned above in the Model file. 

# How to use the files
1. For your image datasets, they should be patched with sizes of 128x128, along with the corresponding masks
2. Follow the folder straucture in the AttResUnet_training.py.

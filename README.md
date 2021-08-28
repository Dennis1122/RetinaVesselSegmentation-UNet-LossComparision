# Evaluation of loss functions for curvilinear structure segmentation
Loss functions play an important part in training a neural network. 
The work aims to compare different loss functions that can be used for the semantic segmentation of retinal vessel images.
This repo contains the implementation of below loss functions and the training setup using U-Net and DRIVE retinal vessel dataset.
1. Binary Cross Entropy
1. Focal loss
1. Tversky loss
1. Focal Tversky loss
1. Dice loss   
1. clDice loss
1. Combined losses
    1. Dice-clDice loss
    1. Tversky-clDice loss
    1. Focal Tversky-clDice loss

## Requiremnts
The project specific requirements can be installed using the below commands. It is recommended to create a 
virtual environment with Python3 as the runtime and install the requirements.

```bash
cd RetinaVesselSegmentation-UNet-LossComparision
pip install -r requirements.txt
```


## Configuration Variable(s)
Configuration variables are stored in `config.py` file in the root directory of this repository. 
Below is a list of the variable(s) which can be modified as needed.

|Name|Required|Description|
|--- |--- |--- |
|EXP_NAME|`true`|Name of experiment (folders are created with this name)|
|loss_function|`true`|Loss function to be used for the experiment|
|train_path|`true`|Path to train images folder|
|test_path|`true`|Path to test images folder|


## Credits
The credits for the loss function implementations goes to these repositories
* [segLoss](https://github.com/JunMa11/SegLoss)
* [clDice](https://github.com/jacobkoenig/clDice-Loss)

# Wide Residual Networks in PyTorch

Implementation of Wide Residual Networks (WRNs) in PyTorch. 

## How to train WRNs

At the moment the CIFAR10 and SVHN datasets are fully supported, with specific augmentations for CIFAR10 drawn from related literature and mean/std normalization for SVHN, and multistep learning rate scheduling for both cases. Training is executed through JSON configuration files, which you can modify or extend to support other configurations of WRNs and/or extend datasets etc.

### Example Runs

Train a WideResNet-16-1 on CIFAR10: 
```
python train.py --config configs/WRN-16-1-scratch-CIFAR10.json
```

Train a WideResNet-40-2 on SVHN: 
```
python train.py --config configs/WRN-40-2-scratch-SVHN.json
```

## Results

This work has been tested with 4 variants of WRNs. When setting the seed generator equal to 0, you should expect a test-set accuracy performance close to the following values:

|Model     | CIFAR10 | SVHN   |
|:---------|:--------|:-------| 
| WRN-16-1 |90.97%   | 95.52% |        
| WRN-16-2 |94.21%   | 96.17% |        
| WRN-40-1 |93.52%   | 96.07% |        
| WRN-40-2 |95.14%   | 96.14% |           

## Notes

The motivation for originally implementing WRNs in PyTorch was [this](https://github.com/AlexandrosFerles/NIPS_2019_Reproducibilty_Challenge_Zero-shot_Knowledge_Transfer_via_Adversarial_Belief_Matching) NeurIPS reproducibility project, where WRNs were used as the main framework for few-shot and zero-shot knowledge transfer. 
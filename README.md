# JNSFANet for No-reference Super-resolution Image Quality Assessment
No-reference super-resolution image quality assessment (NR-SRIQA) is still a challenging task due to complicated degradation factors and the lack of reference high-resolution (HR) images. However, either traditional hand-crafted perceptual features or deep learning-based features are not sufficient to quantify the degradation of SR images. In this paper, we propose a novel NR-SRIQA framework called JNSFANet that integrates natural scene statistics (NNS) features and spatial-frequency mixed domain attention (SFMDA) lightweight network to jointly predict the quality of SR images. The NSS features are extracted by the BRISQUE algorithm through fitting generalized Gaussian distribution (GGD) and asymmetric generalized Gaussian distribution (AGGD) with mean subtracted contrast normalized (MSCN) coefficients. While the spatial-frequency domain attention lightweight network employs MobileNet-V2 as the backbone to reveal non-local authentic distortion features that NSS-based features cannot represent well. By integrating the NNS-based perceptual features and deep learning-based features, the proposed NR-SRIQA model facilitates more compelling quality prediction but requires obviously less computational cost in contrast to compared deep networks for NR-SRIQA. Experimental results verify the superior performance of our method on two benchmark databases in terms of both quantitative evaluations and thorough analysis on the model deployment.

## Model Architecture
![image](https://github.com/kbzhang0505/JNSFANet/blob/main/figures/1.png)
Architecture of the proposed JNSFANet for NR-SRIQA. Convolutional layer notation: Conv(input channels, output channels, kernel size, stride). The detail of NSSFE, SFMDA and IRB layers can be find in our paper.
## Comparison Experiment

## Model Training
Runing trainer.py, the result and weights will be saved in current directory.

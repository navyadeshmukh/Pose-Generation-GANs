Implementation of Deformable GANs on Market-1501 dataset for pose-conditioned human image synthesis.

# Pose Conditioned Human Image Generation using GANs
---

## Objective

The objective of this project is to implement the **Deformable Generative Adversarial Network (GAN)** architecture proposed in the Research Paper *arXiv preprint arXiv:1905.00007* by Siarohin et al. for synthesizing person images conditioned on a specific target pose.

Specifically, given a **source image ($X_a$)** of a person and a **target pose map ($H_b$)**, the model is intended to generate a new, realistic image ($\hat{X}_b$) of that person in the novel pose while preserving the appearance details from $X_a$.

---

## Approach & Implementation Milestones
This project built upon fundamental Machine Learning and Deep Learning concepts, culminating in the final PoseGAN architecture.

### 1. Foundational Deep Learning
* Learned basic concepts of Neural Networks, Machine Learning, optimizers, and hyperparameter tuning.
* **Implemented a handwritten digit recognition model** (MNIST) using 1- and 2-layer Artificial Neural Networks (ANN) with **NumPy from scratch**.

### 2. Advanced Computer Vision & Generative Models
* Studied various Convolutional Neural Network (CNN) architectures.
* Implemented an **object detection model** on the CIFAR-10 dataset using a deep CNN.
* Implemented vanilla **Generative Adversarial Networks (GANs)** and **Conditional GANs (cGANs)** on the MNIST dataset.

### 3. Final Project Implementation (PoseGAN)
* **Implemented the Deformable GAN architecture** in PyTorch, attempting to replicate the key components outlined in the paper *Appearance and Pose-Conditioned Human Image Generation using Deformable GANs* (Siarohin et al.).
* The model was trained on the **Market-1501 dataset**.

---

## Custom Dataset: `Market-1501 Dataset`

This project utilizes a custom PyTorch dataset class, `MarketPoseDataset`, designed for the task using the **Market-1501 dataset**.

### Pairing Strategy
The dataset uses pairs of images of the same person: a **source image ($X_a$)** and a **target image ($X_b$)**.

### Pose Heatmaps
**18 human joint keypoints** are extracted from each image using YOLO-based pose estimation to generate corresponding **pose heatmaps ($H_a, H_b$)**.



### Dataset Item Structure
Each item returned by the dataset is a dictionary containing the following tensors:
1.  **Source image**
2.  **Target image**
3.  **Source pose map**
4.  **Target pose map**

---

## Model Architecture Overview

The core model is a Generative Adversarial Network (GAN) framework, implementing the key components proposed in the reference paper.

### Generator Architecture: Deformable Generator

The Generator ($G$) is a U-Net-based structure designed for pose-guided feature alignment.

| Component | Input | Purpose | Key Concept from Paper |
| :--- | :--- | :--- | :--- |
| **Encoders (Dual Branch)** | Source ($X_a, H_a$) & Target ($H_b$) | Disentangles appearance features from pose features. | Two separate encoders for appearance and pose. |
| **Deformable Warping** | Source Features, $H_a$, $H_b$ | Warps source appearance features to align with the target pose. | **Deformable Skip Connections** |
| **Decoder** | Warped/Aligned Features | Reconstructs the final image ($\hat{X}_b$). | U-Net based reconstruction. |

> **Core Innovation Acknowledged:** The use of **Deformable Skip Connections** is crucial to address the large pixel-to-pixel misalignments caused by significant pose differences, which standard skip connections cannot handle reliably.

### Discriminator Inputs

The Discriminator ($D$) is tasked with distinguishing between:
* **Real Pair:** $(X_b, H_b)$
* **Fake Pair:** $(\hat{X}_b, H_b)$
* **Source Reference:** $(X_a, H_a)$ - *Used to ensure identity preservation.*

---

## Loss Functions

The training utilizes a compound loss function for the Generator ($L_G$), a critical component for achieving high-quality results as outlined in the research.

### 1. Discriminator Loss ($L_D$)
Standard adversarial loss.

### 2. Generator Loss ($L_G$)
A composite loss that encourages realism and identity preservation:

| Sub-Loss | Purpose | Key Concept from Paper |
| :--- | :--- | :--- |
| **Perceptual Loss** | Measures difference in **high-level visual features** to preserve texture and consistency. | Standard practice in GANs to avoid blurriness. |
| **Nearest Neighbour Loss** | Preserves fine texture and structural details by matching local information. | **Proposed in the reference paper** as an alternative to $L_1/L_2$ for better detail transfer despite misalignments. |
| **Offset Smoothness Loss** | Ensures the transformations within the deformable skip connections are smooth. | Ensures spatial coherence and realistic deformation. |

---

## Results

The implementation successfully demonstrates the capability of generating human images conditioned on a target pose, effectively re-posing the person while preserving their appearance.



### Current Results Commentary

The current generated images exhibit some **blurriness and lack of fine detail**. This is primarily attributed to:
* **Insufficient training epochs/data size:** The model requires extensive training on a large dataset to fully converge and capture high-frequency details.
* The training process occasionally **converging to vanishing gradients**, particularly in the later stages of training, which hinders the model's ability to refine the image quality. **Further tuning of the loss weights and optimization parameters is required** to match the performance demonstrated in the reference paper.

---

## Reference

This project is an implementation attempt of the architecture described in:
* **Siarohin, A., Sangineto, E., Lathuilière, S., & Sebe, N. (2019). Appearance and Pose-Conditioned Human Image Generation using Deformable GANs.** *arXiv preprint arXiv:1905.00007.*
    * **arXiv Link:** https://arxiv.org/abs/1905.00007
    * **CVPR 2018 Version (Prior):** *Deformable GANs for Pose-based Human Image Generation.*
 
### Code Repositories Acknowledged:
* **Original Implementation (TensorFlow):** [https://github.com/AliaksandrSiarohin/pose-gan](https://github.com/AliaksandrSiarohin/pose-gan)
* **PyTorch Reference Implementation:** [https://github.com/saurabhsharma1993/pose-transfer](https://github.com/saurabhsharma1993/pose-transfer)

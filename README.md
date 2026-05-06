# LRS-Attack

[[Paper]()] [[Supp]()] [[Poster]()] [[Video]()]

The official implementation of [**\[ICML 2026\] "Low-Rank and Sparsity Are All You Need: Exploring Robust Hierarchical Latent Subspaces for Transferable Adversarial Attack", Shuangshuang Pu, Wen Yang, Min Li, Guodong Liu, Chris Ding, Di Ming*.**]()

## Introduction

Adversarial examples pose serious threats to deep neural networks, exposing fundamental vulnerabilities in model robustness. However, most existing adversarial attacks directly manipulate densely activated and highly redundant feature representations, often leading to overfitting on surrogate models and poor black-box transferability. Recent SVD-based attack attempts to exploit low-rank feature subspaces, yet its reliance on single-layer optimization and single-gradient pathway neglects structural redundancy in feature representations and hierarchical heterogeneity across network layers. To address these limitations, we propose LRS-Attack, a low-rank and sparse decomposition-based attack that explicitly models robust hierarchical subspaces in latent feature spaces. Specifically, the low-rank component captures dominant semantic directions, while the sparse component captures localized and discriminative patterns. To efficiently extract low-rank structure while preserving subspace fidelity, we develop a warm-started alternating low-rank approximation algorithm. Moreover, we introduce a hierarchical mixture of robust experts that leverages depth-dependent feature characteristics and guides gradient optimization toward more transferable adversarial directions. Extensive experiments on ImageNet show that LRS-Attack consistently improves black-box transferability over state-of-the-art methods across diverse CNN/ViT architectures and defense settings.

# 📦 Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation (Top-10 Solutions at ISBI 2025)

**Official Repository:** https://www.codabench.org/competitions/4781/

This repository provides the **complete, browsable source code, trained models, and documentation** for the **Top-10 award-winning methods** from the **Fetal Ultrasound Grand Challenge (FUGC) at ISBI 2025**.  
The challenge focuses on **semi-supervised cervical segmentation in transvaginal ultrasound (TVS)** for **preterm birth risk assessment**, where only **50 expert-labeled images** are available for training.

---

## 🧠 About FUGC 2025

FUGC is the **first international benchmark** dedicated to **semi-supervised cervical segmentation in ultrasound**.  
It provides:

- 50 labeled + 450 unlabeled training images  
- 90 validation images  
- 300 held-out test images  
- Evaluation by **DSC, HD, and runtime (RT)**  
- A weighted ranking scheme: **0.4 × DSC + 0.4 × HD + 0.2 × RT**

**Dataset Repository:** https://zenodo.org/records/14305302

The challenge promotes **human-in-the-loop learning, foundation models, consistency learning, and pseudo-labeling** under extreme label scarcity.

---

## 🏆 Top-10 Team Implementations

| Team | Method | Code | Paper |
|------|--------|------|------|
| T1 | Tran et al. | https://zenodo.org/records/16013666 | Human-in-the-loop semi-supervised uterine cervix ultrasound image segmentation |
| T2 | Pham et al. | https://zenodo.org/records/16014145 | Fetal-bcp: Addressing empirical distribution gap in semi-supervised fetal ultrasound segmentation |
| T3 | Liu et al. | https://zenodo.org/records/16014322 | Light dinov2-unimatch for semi-supervised cervical segmentation |
| T4 | Zhang et al. | https://zenodo.org/records/16014436 | Semi-supervised segmentation of cervical structures in ultrasound images with pseudo-labels and voting strategy |
| T5 | Huang et al. | https://zenodo.org/records/16015993 | Leveraging the semi-supervised learning method unimatch v2 for the fetal ultrasound grand challenge |
| T6 | Chen et al. | https://zenodo.org/records/16016516 | Comt: Co-training mean teachers semi-supervised training framework for cervical segmentation |
| T7 | Liu et al. | https://zenodo.org/records/16016668 | A vision transformer and resnet ensemble method based on the unimatch semi-supervised segmentation framework for the fetal ultrasound grand challenge |
| T8 | Xiao et al. | https://zenodo.org/records/16017032 | Hierarchical augmentation consistency learning for semi-supervised medical image segmentation |
| T9 | Islam et al. | https://zenodo.org/records/16017188 | Semi-supervised transformerbased cervical segmentation: Fugc 2025 challenge |
| T10 | Jiang et al. | https://zenodo.org/records/16017509 | Semi-cervixseg: A multi-stage training strategy for semi-supervised cervical segmentation |

---

## 🔹 T1 – Tran et al. (Human-in-the-Loop U-Net)
Human-in-the-loop semi-supervised framework based on U-Net. Pseudo-labels are iteratively generated and refined by clinicians using Label Studio, then progressively incorporated into training.  
**Code:** https://zenodo.org/records/16013666  

---

## 🔹 T2 – Pham et al. (Fetal-BCP, Mean-Teacher + Copy-Paste)
Bidirectional copy-paste augmentation embedded into a Mean-Teacher framework with Dice+CE supervision and EMA pseudo-labeling.  
**Code:** https://zenodo.org/records/16014145  

---

## 🔹 T3 – Liu et al. (UniMatch-V2 + DINOv2-S)
Consistency-based semi-supervised learning using UniMatch-V2 with a DINOv2-Small ViT backbone.  
**Code:** https://zenodo.org/records/16014322  

---

## 🔹 T4 – Zhang et al. (Two-Stage nnU-Net + U-Net)
nnU-Net generates pseudo-labels; U-Net retrains on weighted labeled+pseudo-labeled data with ensemble refinement.  
**Code:** https://zenodo.org/records/16014436  

---

## 🔹 T5 – Huang et al. (UniMatch-V2 + DINOv2)
Single-stream UniMatch with DINOv2 backbone, EMA teachers, and channel-wise dropout.  
**Code:** https://zenodo.org/records/16015993  

---

## 🔹 T6 – Chen et al. (Dual-Teacher Co-Training)
U-Net and DeepLabV3 co-teaching with entropy-based pseudo-label filtering; fastest runtime.  
**Code:** https://zenodo.org/records/16016516  

---

## 🔹 T7 – Liu et al. (Ensemble UniMatch)
PVTv2-B1 and ResNet34D ensembles trained under UniMatch with pseudo-label refinement.  
**Code:** https://zenodo.org/records/16016668  

---

## 🔹 T8 – Xiao et al. (Copy-Paste Consistency Learning)
Teacher–student SSL using copy-paste augmentation and consistency loss.  
**Code:** https://zenodo.org/records/16017032  

---

## 🔹 T9 – Islam et al. (Transformer-based UniMatch-V2)
UniMatch-V2 with DINOv2-B ViT backbone, feature-level dropout, and multi-scale fusion.  
**Code:** https://zenodo.org/records/16017188  

---

## 🔹 T10 – Jiang et al. (Semi-CervixSeg)
Multi-stage pseudo-label refinement with RWKV-UNet and PVT-EMCAD-B2 using consistency and contrastive learning.  
**Code:** https://zenodo.org/records/16017509  

---

# Diabetic Retinopathy — AI Diagnostic Pipeline
**Capstone Group 12**

This project builds an end-to-end pipeline for automated Diabetic Retinopathy (DR) analysis using deep learning on retinal fundus images.

---

## Pipeline Overview

**1. DR Grading (Classification)**
EfficientNet-B0 trained on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset to classify retinal images into 5 DR severity levels: No DR, Mild, Moderate, Severe, and Proliferative DR. Includes an image quality check (blur/noise detection) before inference.

**2. Lesion Segmentation (U-Net)**
EfficientNet-B0 encoder + U-Net decoder trained on the [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) to segment 5 lesion types: Microaneurysms (MA), Haemorrhages (HEM), Hard Exudates (HE), Soft Exudates (SE), and Optic Disc (OD). Trained with BCE + Dice loss and differential learning rates.

**3. Explainability (XAI)**
GradCAM and SmoothGrad applied to both models to visualize which retinal regions drive classification and segmentation predictions.

---

## Datasets

| Dataset | Used For |
|---|---|
| [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) | DR severity classification |
| [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | Lesion segmentation — 54 train / 27 test images with pixel-level masks |

---

## Models

| Model | Task | Backbone |
|---|---|---|
| `efficientnet_b0_best.pth` | DR Grading | EfficientNet-B0 |
| `unet_effb0_lesion_final.pth` | Lesion Segmentation | EfficientNet-B0 + U-Net |

---

## Key Dependencies
```bash
torch torchvision segmentation-models-pytorch albumentations opencv-python scikit-learn google-generativeai
```

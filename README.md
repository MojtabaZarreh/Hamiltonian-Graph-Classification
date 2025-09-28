# 🧠 Hamiltonian Graph Classification

This repository contains a deep learning model that classifies whether a given **graph image** represents a **Hamiltonian graph** (i.e., a graph that contains a Hamiltonian cycle).

## 📘 What is a Hamiltonian Graph?
A **Hamiltonian graph** is a type of graph that contains a **Hamiltonian cycle** 
a closed loop that visits each vertex **exactly once** before returning to the starting point.
Determining whether a graph is Hamiltonian is an **NP-complete problem**, meaning it’s computationally hard to solve for large graphs.  
This project explores a **visual deep learning approach** to predict Hamiltonian properties **directly from images of graphs**, rather than using traditional graph algorithms.

<img width="886" height="504" alt="download (1)" src="https://github.com/user-attachments/assets/800b785f-5ddc-4c4a-a742-56243aaec814" />

---

## 🚀 Overview
The project uses a **Vision Transformer (ViT)** model to perform **binary classification** on graph images.  
Each image corresponds to a graph, and the label indicates whether it contains a Hamiltonian cycle (`1`) or not (`0`).

---

## 📂 Dataset
The dataset (`hamiltonian_graph_init.zip`) includes:

- `train_images/` — graph images used for training  
- `test_images/` — graph images used for testing  
- `train_info.csv` — CSV file containing file names and binary labels  

The notebook automatically downloads the dataset from Google Drive using `gdown`.

---

## 🧩 Model Architecture
The model uses a **Vision Transformer (ViT)** from the [`timm`](https://github.com/huggingface/pytorch-image-models) library:

```python
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1)

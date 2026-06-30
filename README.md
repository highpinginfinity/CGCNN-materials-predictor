# Crystal Graph Convolutional Neural Networks (CGCNN)

A modernized, lightweight implementation of Crystal Graph Convolutional Neural Networks (CGCNN). This toolkit is designed to predict material properties directly from crystal structures by mapping crystallographic data into mathematical graphs.

This repository is optimized for analyzing functional materials. It can be utilized for advanced structural predictions, such as modeling the structural and energetic effects of specific dopants (e.g., Niobium or Manganese) within complex perovskite crystal lattices like Barium Titanate (BaTiO3).

## Theoretical Background
Traditional machine learning struggles with the periodic nature of crystals. This framework solves that by representing crystal structures as undirected graphs:
* **Nodes:** Represent atoms, initialized with vectors based on their elemental properties (electronegativity, group number, etc.).
* **Edges:** Represent chemical bonds, weighted by interatomic distances using a Gaussian expansion filter.

By passing this graph through sequential convolution layers, the network learns the spatial and chemical environment of every atom, allowing it to accurately predict macroscopic functional properties.

## Features
* **Simplified Data Pipeline:** Direct conversion from `pymatgen` CIF objects to PyTorch Tensors.
* **Modernized Neural Network:** Cleaned-up node and edge convolution layers written in standard PyTorch modules.
* **Flexible Inference:** Easily point the model to a directory of `.cif` files to output predictive properties.

## Usage

### 1. Structure your dataset
Create a folder (e.g., `data/`) containing:
1. Your target `.cif` files.
2. The `atom_init.json` file.
3. An `id_prop.csv` file mapping the CIF IDs to target values.

### 2. Train a model
Train the network on your customized dataset:
```bash
python main.py --mode train --data_dir ./data
```

### 3. Predict properties
Run inference on new, unseen crystal structures using your trained weights:
```bash
python main.py --mode predict --data_dir ./data --weights cgcnn_weights.pth
```
# Structure-Property Correlations Predictor (via Graph Convolutions)

A modernized, lightweight implementation of Crystal Graph Convolutional Neural Networks (CGCNN) designed to predict material properties directly from crystal structures.

This toolkit represents crystal structures as graphs (atoms as nodes, bonds as edges) to map structure-property correlations in functional materials, bridging the gap between raw crystallographic data and functional predictions.

## Features
* **Simplified Data Pipeline:** Direct `pymatgen` to PyTorch Tensor conversion.
* **Modernized Neural Network:** Cleaned-up node and edge convolution layers written in standard PyTorch modules.
* **Plug-and-Play Inference:** Easily point the model to a directory of `.cif` files to output predictive properties.

## Usage

### 1. Structure your dataset
Create a folder (e.g., `data/`) containing your `.cif` files, an `atom_init.json` file, and an `id_prop.csv` file mapping the CIF IDs to target values.

### 2. Train a model
```bash
python main.py --mode train --data_dir ./data
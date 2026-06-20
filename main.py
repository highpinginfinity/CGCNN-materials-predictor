import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CIFDataset
from model import SimpleCGCNN
import argparse

def train_model(data_dir, epochs=50):
    dataset = CIFDataset(data_dir)
    model = SimpleCGCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for atom_fea, nbr_fea, nbr_fea_idx, target, _ in dataset:
            optimizer.zero_grad()
            output = model(atom_fea, nbr_fea, nbr_fea_idx)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}")
    
    torch.save(model.state_dict(), "cgcnn_weights.pth")
    print("Model saved to cgcnn_weights.pth")

def predict(data_dir, weights_path):
    dataset = CIFDataset(data_dir)
    model = SimpleCGCNN()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    print("ID, Predicted Value")
    with torch.no_grad():
        for atom_fea, nbr_fea, nbr_fea_idx, target, cif_id in dataset:
            output = model(atom_fea, nbr_fea, nbr_fea_idx)
            print(f"{cif_id}, {output.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified CGCNN")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--data_dir', required=True, help="Path to CIF dataset")
    parser.add_argument('--weights', help="Path to weights for prediction", default="cgcnn_weights.pth")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.data_dir)
    else:
        predict(args.data_dir, args.weights)
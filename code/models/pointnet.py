import os
import csv
import random
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.transforms import NormalizeScale, SamplePoints
from torch_geometric.typing import WITH_TORCH_CLUSTER
from sklearn.model_selection import KFold

WITH_TORCH_CLUSTER = True  # Force recognition of torch-cluster

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
# Function to extract point cloud from segmentation file
def extract_point_cloud_from_segmentation(segmentation_file, label_ids):
    img = nib.load(segmentation_file)
    data = img.get_fdata()

    # Extract points corresponding to the label IDs
    points = []
    for label_id in label_ids:
        coords = np.argwhere(data == label_id)
        points.append(coords)

    # Combine points from all specified labels
    points = np.vstack(points)
    return points

# Function for data augmentation
def augment_point_cloud(points):
    # Random Rotation
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,              0,             1]])
    points = np.dot(points, rotation_matrix)

    # Random Translation
    translation = np.random.uniform(-0.2, 0.2, size=(1, 3))
    points += translation

    # Random Scaling
    scale = np.random.uniform(0.8, 1.2)
    points *= scale

    return points

# Define the PointNet model architecture
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.mlp = MLP([1024, 512, 256, 2], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1)

# Function to load data
def load_data(mnd_folder, normal_folder):
    data = []

    # Iterate over MND folders
    for i in range(1, len(os.listdir(mnd_folder)) + 1):
        stats_folder = os.path.join(mnd_folder, str(i), 'stats')
        mri_folder = os.path.join(mnd_folder, str(i), 'mri')

        if os.path.exists(stats_folder) and os.path.exists(mri_folder):
            data.append((mri_folder, stats_folder, 1))  # Label 1 for MND patients

    # Iterate over Normal Control folders
    for i in range(1, len(os.listdir(normal_folder)) + 1):
        stats_folder = os.path.join(normal_folder, str(i), 'stats')
        mri_folder = os.path.join(normal_folder, str(i), 'mri')

        if os.path.exists(stats_folder) and os.path.exists(mri_folder):
            data.append((mri_folder, stats_folder, 0))  # Label 0 for Normal Controls

    return data

# Function to create DataLoader from the dataset
def create_dataloader(data, batch_size=16, augment=False):
    dataset = []
    for mri_folder, stats_folder, label in data:
        # Get the segmentation file
        segmentation_file = os.path.join(mri_folder, 'aparc.DKTatlas+aseg.deep.withCC.mgz')
        point_cloud = extract_point_cloud_from_segmentation(segmentation_file, precentral_gyrus_labels)

        # Apply data augmentation if required
        if augment:
            point_cloud = augment_point_cloud(point_cloud)

        # Convert the point cloud to PyTorch tensors
        pos = torch.tensor(point_cloud, dtype=torch.float)
        batch = torch.zeros(pos.shape[0], dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(pos=pos, batch=batch, y=y)
        dataset.append(data)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Train and Test the Model
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, data.y, reduction='sum').item()
            pred = output.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return total_loss / len(val_loader.dataset), correct / len(val_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data).max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    # Paths
    mnd_folder = './MND_patients'
    normal_folder = './Normal_control'

    # Define the segmentation file and label IDs for precentral gyrus
    precentral_gyrus_labels = [1024, 2024]  # Left and right precentral gyrus in Desikan-Killiany atlas

    # Load data
    data = load_data(mnd_folder, normal_folder)

    # Implement K-fold cross-validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Split data
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        # Create DataLoaders
        train_loader = create_dataloader(train_data, augment=True)  # Enable augmentation for training data
        val_loader = create_dataloader(val_data)

        # Training loop
        for epoch in range(1, 201):
            train_loss = train(model, train_loader, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, device)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Test the model after training on each fold
        test_loader = create_dataloader(val_data)  # Use the validation set of this fold as the test set
        test_acc = test(model, test_loader, device)
        print(f'Test Accuracy for fold {fold + 1}: {test_acc:.4f}')

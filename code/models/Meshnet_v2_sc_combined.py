import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel.freesurfer.io as fsio
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import trimesh
from torch_scatter import scatter_mean  # Ensure torch_scatter is installed
import warnings
import seaborn as sns  # For improved plotting styles

# Print versions and availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print("MeshNet v2 Official Implementation Started (Combined Precentral + Paracentral)")
print(f"Trimesh version: {trimesh.__version__}")

# Suppress specific deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`simplify_quadratic_decimation` is deprecated as it was a typo and will be removed in March 2024: replace with `simplify_quadric_decimation`",
)

# Function to simplify mesh using Trimesh
def simplify_mesh(vertices, faces, target_faces=1024):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) > target_faces:
        simplified_mesh = mesh.simplify_quadric_decimation(target_faces)  # Updated to the correct method
        return simplified_mesh.vertices, simplified_mesh.faces
    return vertices, faces

# Function to extract and merge multiple regions from .pial and .annot files
def extract_merged_region_from_pial(pial_file, annot_file, target_region_names):
    """
    Extract and merge the surface meshes of multiple regions from the .pial file based on the annotation.

    Parameters:
    pial_file (str): Path to the .pial file.
    annot_file (str): Path to the .annot file.
    target_region_names (list): List of target region names to merge.

    Returns:
    tuple: Merged vertices and faces of the target regions.
    """
    # Load the pial file
    vertices, faces = fsio.read_geometry(pial_file)
    faces = faces.astype(np.int64)

    # Load the annotation file
    labels, ctab, names = fsio.read_annot(annot_file)
    # Decode the region names
    names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in names]

    # Get label indices for target regions
    target_label_indices = [names.index(region_name) for region_name in target_region_names if region_name in names]
    if not target_label_indices:
        print(f"Regions {target_region_names} not found in annotation file")
        return None, None

    # Create a mask for vertices belonging to any of the target regions
    region_mask = np.isin(labels, target_label_indices)

    # Extract the indices of vertices that belong to the target regions
    region_indices = np.where(region_mask)[0]

    if len(region_indices) == 0:
        print(f"No vertices found for regions {target_region_names} in {pial_file}")
        return None, None

    # Create a map from original indices to new indices for the target regions
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(region_indices)}

    # Extract the vertices for the target regions
    region_vertices = vertices[region_indices]

    # Find faces that are composed of vertices all within the target regions
    valid_faces = []
    for face in faces:
        if all(vertex in index_map for vertex in face):
            mapped_face = [index_map[vertex] for vertex in face]
            valid_faces.append(mapped_face)

    if len(valid_faces) == 0:
        print(f"No faces found for regions {target_region_names} in {pial_file}")
        return None, None

    region_faces = np.array(valid_faces)

    return region_vertices, region_faces

# Function for data augmentation
def augment_vertices(vertices):
    # Random Rotation around Z-axis
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ]
    )
    vertices = np.dot(vertices, rotation_matrix)

    # Random Translation
    translation = np.random.uniform(-0.2, 0.2, size=(1, 3))
    vertices += translation

    # Random Scaling
    scale = np.random.uniform(0.8, 1.2)
    vertices *= scale

    # Add Gaussian noise
    noise = np.random.normal(0, 0.02, vertices.shape)
    vertices += noise

    return vertices

# Function to compute face features and edge indices
def compute_face_features_and_edge_index(pos, face):
    """
    Compute spatial and structural features for each face.

    Parameters:
    pos (torch.Tensor): Vertex positions, shape [num_nodes, 3].
    face (torch.Tensor): Face indices, shape [3, num_faces].

    Returns:
    tuple: face_features [num_faces, 16], edge_index [2, num_edges].
    """
    # pos: [num_nodes, 3]
    # face: [3, num_faces]
    v0 = pos[face[0]]  # [num_faces, 3]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    # Compute face normals
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = torch.norm(normals, dim=1, keepdim=True) / 2.0
    normals = F.normalize(normals, p=2, dim=1)

    # Compute face centers
    centers = (v0 + v1 + v2) / 3.0

    # Compute corner features (flattened vertex positions of each face)
    corners = torch.cat([v0, v1, v2], dim=1)  # [num_faces, 9]

    # Stack spatial features
    face_features = torch.cat([normals, corners, centers, areas], dim=1)  # Shape: [num_faces, 16]

    # Build edge indices for adjacency
    num_faces = face.shape[1]
    face_np = face.cpu().numpy().T  # [num_faces, 3]

    # Build a mapping from vertex to faces
    vertex_to_faces = {}
    for i, f in enumerate(face_np):
        for vertex in f:
            if vertex not in vertex_to_faces:
                vertex_to_faces[vertex] = []
            vertex_to_faces[vertex].append(i)

    # For each face, find adjacent faces
    row = []
    col = []
    for i in range(num_faces):
        adjacent_faces = set()
        for vertex in face_np[i]:
            adjacent_faces.update(vertex_to_faces[vertex])
        adjacent_faces.discard(i)  # Remove the face itself
        for adj_face in adjacent_faces:
            row.append(i)
            col.append(adj_face)

    # Create edge_index tensor
    edge_index = torch.tensor([row, col], dtype=torch.long)  # Shape: [2, num_edges]

    # Compute structural features (mean dihedral angle per face)
    struct_features = compute_structural_features(normals, edge_index, num_faces)

    # Concatenate spatial and structural features
    face_features = torch.cat([face_features, struct_features], dim=1)  # Shape: [num_faces, 17]

    return face_features, edge_index

# Function to compute structural features
def compute_structural_features(normals, edge_index, num_faces):
    """
    Compute structural features (mean dihedral angle) for each face.

    Parameters:
    normals (torch.Tensor): Face normals, shape [num_faces, 3].
    edge_index (torch.Tensor): Adjacency edges, shape [2, num_edges].
    num_faces (int): Number of faces.

    Returns:
    torch.Tensor: Structural features, shape [num_faces, 1].
    """
    # normals: [num_faces, 3]
    source = edge_index[0]
    target = edge_index[1]
    normal_source = normals[source]  # [num_edges, 3]
    normal_target = normals[target]  # [num_edges, 3]

    # Compute dihedral angles
    dot_product = (normal_source * normal_target).sum(dim=1).clamp(-1.0, 1.0)  # [num_edges]
    angles = torch.acos(dot_product)  # [num_edges]

    # Assign angles to both source and target faces
    angles = angles.repeat(2)
    faces = torch.cat([source, target], dim=0)  # [2 * num_edges]

    # Initialize structural features
    struct_features = torch.zeros((num_faces, 1), device=angles.device)

    # Compute mean angle per face using scatter_mean
    struct_features = scatter_mean(angles.unsqueeze(1), faces, dim=0, dim_size=num_faces)

    # Handle faces with no adjacent faces (if any)
    struct_features[struct_features != struct_features] = 0.0  # Replace NaN with 0

    return struct_features  # Shape: [num_faces, 1]

# Function to load data
def load_data(mnd_folder, normal_folder):
    data = []
    labels = []

    # Load MND patients
    for i in os.listdir(mnd_folder):
        if i == str(108):  # Skipping subject '108' as per original code
            continue
        subject_folder = os.path.join(mnd_folder, i)
        surf_folder = os.path.join(subject_folder, 'surf')
        label_folder = os.path.join(subject_folder, 'label')
        if os.path.exists(surf_folder) and os.path.exists(label_folder):
            data.append((surf_folder, label_folder))
            labels.append(1)  # Label 1 for MND patients

    # Load Normal Controls
    for i in os.listdir(normal_folder):
        subject_folder = os.path.join(normal_folder, i)
        surf_folder = os.path.join(subject_folder, 'surf')
        label_folder = os.path.join(subject_folder, 'label')
        if os.path.exists(surf_folder) and os.path.exists(label_folder):
            data.append((surf_folder, label_folder))
            labels.append(0)  # Label 0 for Normal Controls

    return data, labels

# Function to create DataLoader from the dataset
def create_dataloader(
    data,
    labels,
    batch_size=32,
    augment=False,
    target_faces=1024,
    target_region_names=['precentral', 'paracentral'],
):
    """
    Create a DataLoader for batched mesh data with merged regions.

    Parameters:
    data (list): List of tuples containing surf_folder and label_folder paths.
    labels (list): List of labels corresponding to the data.
    batch_size (int): Number of samples per batch.
    augment (bool): Whether to apply data augmentation.
    target_faces (int): Target number of faces per hemisphere after simplification.
    target_region_names (list): List of region names to extract and merge.

    Returns:
    DataLoader: PyTorch Geometric DataLoader object.
    """
    dataset = []
    for (surf_folder, label_folder), label in zip(data, labels):
        # Process both hemispheres
        hemispheres = ['lh', 'rh']
        all_vertices = []
        all_faces = []
        offset = 0  # To ensure face indices are correct when combining hemispheres

        for hemi in hemispheres:
            pial_file = os.path.join(surf_folder, f'{hemi}.pial')
            annot_file = os.path.join(label_folder, f'{hemi}.aparc.DKTatlas.annot')
            if os.path.exists(pial_file) and os.path.exists(annot_file):
                vertices, faces = extract_merged_region_from_pial(
                    pial_file, annot_file, target_region_names
                )
                if vertices is None or faces is None:
                    continue  # Skip if no region found
                # Simplify the mesh
                vertices, faces = simplify_mesh(vertices, faces, target_faces=target_faces)
                # If augment is True, apply augmentation to vertices
                if augment:
                    vertices = augment_vertices(vertices)
                # Adjust face indices when combining hemispheres
                faces = faces + offset
                offset += len(vertices)
                all_vertices.append(vertices)
                all_faces.append(faces)
            else:
                print(f'Missing pial or annot file for {surf_folder}')

        if len(all_vertices) == 0:
            continue  # Skip this subject if no data

        # Combine vertices and faces from all hemispheres
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)

        # Enforce uniform number of faces per graph
        desired_faces = target_faces * 2  # 2 hemispheres
        current_faces = faces.shape[0]

        if current_faces < desired_faces:
            # Pad faces by repeating the last face
            pad_faces = np.tile(faces[-1], (desired_faces - current_faces, 1))
            faces = np.vstack([faces, pad_faces])
        elif current_faces > desired_faces:
            # Truncate faces to desired_faces
            faces = faces[:desired_faces]

        # Prepare data for torch_geometric
        pos = torch.tensor(vertices, dtype=torch.float)
        face = torch.tensor(faces.T, dtype=torch.long)  # Transpose to shape [3, num_faces]
        y = torch.tensor([label], dtype=torch.long)

        # Compute face features and edge indices
        face_features, edge_index = compute_face_features_and_edge_index(pos, face)

        # Enforce uniform number of face_features
        if face_features.size(0) < desired_faces:
            # Pad face_features by repeating the last feature
            pad_size = desired_faces - face_features.size(0)
            pad_features = face_features[-1].unsqueeze(0).repeat(pad_size, 1)
            face_features = torch.cat([face_features, pad_features], dim=0)
        elif face_features.size(0) > desired_faces:
            # Truncate face_features
            face_features = face_features[:desired_faces]

        # Create Data object
        data_obj = Data(
            pos=pos,
            face=face,
            face_features=face_features,
            edge_index=edge_index,
            y=y,
        )
        dataset.append(data_obj)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Structural Descriptor: Face Kernel Correlation and Face Rotate Convolution
class StructuralDescriptor(nn.Module):
    def __init__(self):
        super(StructuralDescriptor, self).__init__()
        self.fc_correlation = nn.Conv1d(3, 64, 1)  # Face Kernel Correlation (normal vectors)
        self.fc_rotation = nn.Conv1d(9, 64, 1)      # Face Rotate Convolution (corner features)

    def forward(self, normal, corner):
        correlation_features = F.relu(self.fc_correlation(normal))
        rotation_features = F.relu(self.fc_rotation(corner))
        features = torch.cat([correlation_features, rotation_features], dim=1)  # [batch_size, 128, num_faces]
        return features

# Spatial Descriptor: Center coordinate-based MLP
class SpatialDescriptor(nn.Module):
    def __init__(self):
        super(SpatialDescriptor, self).__init__()
        self.mlp = nn.Conv1d(3, 64, 1)

    def forward(self, center):
        spatial_features = F.relu(self.mlp(center))  # [batch_size, 64, num_faces]
        return spatial_features

# Official MeshNet implementation (single channel: combined precentral + paracentral)
class MeshNetOfficial(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(MeshNetOfficial, self).__init__()
        # Structural and Spatial Descriptors
        self.structural_descriptor = StructuralDescriptor()
        self.spatial_descriptor = SpatialDescriptor()

        # Convolution layers
        self.conv1 = nn.Conv1d(192, 256, 1)  # 128 (structural) + 64 (spatial) = 192
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)

        # Global pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, data):
        batch_size = data.y.size(0)
        face_features = data.face_features  # [batch_size*num_faces, 17]

        # Split face_features into components
        normals = face_features[:, :3]        # [batch_size*num_faces, 3]
        corners = face_features[:, 3:12]      # [batch_size*num_faces, 9]
        centers = face_features[:, 12:15]     # [batch_size*num_faces, 3]
        areas = face_features[:, 15:16]       # [batch_size*num_faces, 1]
        struct_features = face_features[:, 16:17]  # [batch_size*num_faces, 1]

        # Reshape and transpose to [batch_size, num_features, num_faces]
        num_faces = normals.size(0) // batch_size
        normal = normals.view(batch_size, num_faces, -1).transpose(1, 2)    # [batch_size, 3, num_faces]
        corner = corners.view(batch_size, num_faces, -1).transpose(1, 2)    # [batch_size, 9, num_faces]
        center = centers.view(batch_size, num_faces, -1).transpose(1, 2)    # [batch_size, 3, num_faces]

        # Structural and Spatial descriptors
        struct_features = self.structural_descriptor(normal, corner)  # [batch_size, 128, num_faces]
        spatial_features = self.spatial_descriptor(center)            # [batch_size, 64, num_faces]

        combined_features = torch.cat([struct_features, spatial_features], dim=1)  # [batch_size, 192, num_faces]

        x = F.relu(self.bn1(self.conv1(combined_features)))  # [batch_size, 256, num_faces]
        x = F.relu(self.bn2(self.conv2(x)))                # [batch_size, 512, num_faces]
        x = F.relu(self.bn3(self.conv3(x)))                # [batch_size, 512, num_faces]
        x = self.pool(x).squeeze(-1)                        # [batch_size, 512]

        x = F.relu(self.bn4(self.fc1(x)))                   # [batch_size, 1024]
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))                   # [batch_size, 512]
        x = self.dropout2(x)
        x = self.fc3(x)                                      # [batch_size, num_classes]

        return x

# Train function
def train(model, train_loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y.to(device), weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validate function
def validate(model, val_loader, device, class_weights):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = F.cross_entropy(out, data.y.to(device), weight=class_weights)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(val_loader), acc, prec, rec, all_preds, all_labels

# Exponential smoothing function
def exponential_smoothing(values, alpha=0.9):
    smoothed = []
    for i in range(len(values)):
        if i == 0:
            smoothed.append(values[0])
        else:
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed

# Plot training curves
def plot_training_curves(metrics_dict, hyperparams):
    epochs = range(1, len(metrics_dict['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    sns.set_style("whitegrid")

    # Loss
    axs[0, 0].plot(epochs, metrics_dict['train_loss'], label='Train Loss', color='blue')
    axs[0, 0].plot(epochs, metrics_dict['val_loss'], label='Val Loss', color='red')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Accuracy
    axs[0, 1].plot(epochs, metrics_dict['val_acc'], label='Val Accuracy', color='green')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Precision
    axs[1, 0].plot(epochs, metrics_dict['val_prec'], label='Val Precision', color='purple')
    axs[1, 0].set_title('Validation Precision')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()

    # Recall
    axs[1, 1].plot(epochs, metrics_dict['val_rec'], label='Val Recall', color='orange')
    axs[1, 1].set_title('Validation Recall')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].legend()

    fig.suptitle(f"Training Curves (lr={hyperparams[0]}, dropout={hyperparams[1]}, batch_size={hyperparams[2]})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        f"./models/meshnet_v2_single_channel_combined/training_curves/training_curves_lr{hyperparams[0]}_dropout{hyperparams[1]}_batch_size_{hyperparams[2]}.png"
    )
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(conf_mt_counter, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix {conf_mt_counter}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'./models/meshnet_v2_single_channel_combined/confusion_matrix/confusion_matrix_{conf_mt_counter}.png')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    mnd_folder = './MND_patients'
    normal_folder = './Normal_control'

    # Create directories for saving models and figures
    os.makedirs('./models/meshnet_v2_single_channel_combined/confusion_matrix', exist_ok=True)
    os.makedirs('./models/meshnet_v2_single_channel_combined/hyperparameters', exist_ok=True)
    os.makedirs('./models/meshnet_v2_single_channel_combined/training_curves', exist_ok=True)

    # Load data
    print("Loading data...")
    data, labels = load_data(mnd_folder, normal_folder)
    print(f"Total samples loaded: {len(data)}")

    # Split data into train and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")

    # Hyperparameter search
    learning_rates = [0.001, 0.0001, 0.00001]
    dropout_rates = [0.0, 0.3, 0.5]
    batch_sizes = [64, 128]
    target_faces = 1024  # Adjusted target_faces per hemisphere to enforce uniformity

    best_val_acc = 0
    best_hyperparams = None
    results = []
    conf_mt_counter = 0
    all_metrics = []

    total_combinations = len(learning_rates) * len(dropout_rates) * len(batch_sizes)
    print(f"Total hyperparameter combinations: {total_combinations}")

    for lr, dropout, batch_size in product(learning_rates, dropout_rates, batch_sizes):
        print(f"\nTraining with lr={lr}, dropout={dropout}, batch_size={batch_size}, augment=False")

        # Create DataLoaders with merged regions
        train_loader = create_dataloader(
            train_data,
            train_labels,
            batch_size=batch_size,
            augment=False,
            target_faces=target_faces,
            target_region_names=['precentral', 'paracentral'],
        )
        val_loader = create_dataloader(
            val_data,
            val_labels,
            batch_size=batch_size,
            augment=False,
            target_faces=target_faces,
            target_region_names=['precentral', 'paracentral'],
        )

        # Initialize model, optimizer, and learning rate scheduler
        model = MeshNetOfficial(num_classes=2, dropout_rate=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training loop
        metrics_dict = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_prec': [],
            'val_rec': [],
        }
        for epoch in tqdm(range(1, 101), desc=f"LR={lr}, Dropout={dropout}, Batch Size={batch_size}"):
            train_loss = train(model, train_loader, optimizer, device, class_weights)
            val_loss, val_acc, val_prec, val_rec, val_preds, val_labels_epoch = validate(
                model, val_loader, device, class_weights
            )
            scheduler.step(val_loss)
            metrics_dict['train_loss'].append(train_loss)
            metrics_dict['val_loss'].append(val_loss)
            metrics_dict['val_acc'].append(val_acc)
            metrics_dict['val_prec'].append(val_prec)
            metrics_dict['val_rec'].append(val_rec)
            print(
                f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}'
            )

        # Plot training curves
        plot_training_curves(metrics_dict, hyperparams=(lr, dropout, batch_size))

        # Plot confusion matrix after last epoch
        plot_confusion_matrix(
            conf_mt_counter, val_labels_epoch, val_preds, classes=['Normal', 'MND']
        )
        conf_mt_counter += 1

        # Record results
        results.append((lr, dropout, batch_size, metrics_dict['val_acc'][-1]))
        all_metrics.append(
            {
                'lr': lr,
                'dropout': dropout,
                'batch_size': batch_size,
                'metrics': metrics_dict,
            }
        )

        # Save best model
        if metrics_dict['val_acc'][-1] > best_val_acc:
            best_val_acc = metrics_dict['val_acc'][-1]
            best_hyperparams = (lr, dropout, batch_size)
            torch.save(model.state_dict(), './models/meshnet_v2_single_channel_combined/best_model.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # Plot hyperparameter search results
    plt.figure(figsize=(12, 8))
    markers = ['o-', 's-', '^-']
    for idx, lr in enumerate(learning_rates):
        lr_results = [r for r in results if r[0] == lr]
        if not lr_results:
            continue
        x = range(1, len(lr_results) + 1)
        y = [r[3] for r in lr_results]
        plt.plot(x, y, markers[idx % len(markers)], label=f'lr={lr}')
        # Apply exponential smoothing
        y_smoothed = exponential_smoothing(y, alpha=0.9)
        plt.plot(x, y_smoothed, '-', label=f'lr={lr} Smoothed')

    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Validation Accuracy')
    plt.title('Hyperparameter Search Results')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./models/meshnet_v2_single_channel_combined/hyperparameters/search_results.png')
    plt.show()

    print(
        f"Best hyperparameters: lr={best_hyperparams[0]}, dropout={best_hyperparams[1]}, batch_size={best_hyperparams[2]}"
    )
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save all metrics for analysis
    torch.save(all_metrics, './models/meshnet_v2_single_channel_combined/hyperparameters/all_metrics.pt')

    print("Training and testing completed.")

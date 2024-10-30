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
import warnings
import nibabel as nib
from skimage import measure

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print("MeshNet multi-channel with BA regions and corpus callosum using Separate Convolutional Paths started")
print(f"Trimesh version: {trimesh.__version__}")

# Suppress the specific deprecation warning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`simplify_quadratic_decimation` is deprecated as it was a typo and will be removed in March 2024: replace with `simplify_quadric_decimation`",
)

# Function to simplify mesh using Trimesh
def simplify_mesh(vertices, faces, target_faces=1024):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) > target_faces:
        simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
        return simplified_mesh.vertices, simplified_mesh.faces
    return vertices, faces

# Function to extract regions from .pial and .label files
def extract_regions_from_pial_using_labels(pial_file, label_files):
    """
    Extract the surface mesh of specific regions from the .pial file based on label files.

    Parameters:
    pial_file (str): Path to the .pial file.
    label_files (list of str): Paths to the .label files for the regions.

    Returns:
    tuple: Extracted vertices and faces for the target regions.
    """
    vertices, faces = fsio.read_geometry(pial_file)
    faces = faces.astype(np.int64)

    # Read all label files and combine the vertex indices
    combined_vertex_indices = np.array([], dtype=np.int64)
    for label_file in label_files:
        if os.path.exists(label_file):
            label_indices = fsio.read_label(label_file)
            combined_vertex_indices = np.union1d(combined_vertex_indices, label_indices)
        else:
            print(f"Label file {label_file} not found.")
            continue

    if len(combined_vertex_indices) == 0:
        print(f"No vertices found for the given label files in {pial_file}")
        return None, None

    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(combined_vertex_indices)}
    region_vertices = vertices[combined_vertex_indices]

    valid_faces = []
    for face in faces:
        if all(vertex in index_map for vertex in face):
            mapped_face = [index_map[vertex] for vertex in face]
            valid_faces.append(mapped_face)

    if len(valid_faces) == 0:
        print(f"No faces found for the given label files in {pial_file}")
        return None, None

    region_faces = np.array(valid_faces)
    return region_vertices, region_faces

# Function to extract corpus callosum regions from ASEG file
def extract_cc_from_aseg(segmentation_file, label_ids, apply_affine=True):
    """
    Extracts the outer surface points and faces for specified label IDs from a segmentation file.

    Parameters:
    segmentation_file (str): Path to the segmentation file.
    label_ids (list of int): List of label IDs to extract.

    Returns:
    tuple: Arrays of vertices and faces.
    """
    # Load the segmentation file
    img = nib.load(segmentation_file)
    data = img.get_fdata()
    affine = img.affine  # Affine transformation matrix

    # Create a mask for the specified labels
    mask = np.isin(data, label_ids)

    # Perform marching cubes to obtain the surface
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

    # Apply affine transformation to vertices if required
    if apply_affine:
        # Add a column of ones to convert to homogeneous coordinates
        verts_homogeneous = np.hstack([verts, np.ones((verts.shape[0], 1))])
        # Apply the affine matrix
        verts_world = verts_homogeneous @ affine.T
        verts = verts_world[:, :3]

    return verts, faces

# Function for data augmentation
def augment_vertices(vertices):
    # Random Rotation
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
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

# Function to compute face features
def compute_face_features(pos, face):
    """
    Compute normal vectors, corner features, and centers for each face.

    Parameters:
    pos (torch.Tensor): Vertex positions, shape [num_nodes, 3].
    face (torch.Tensor): Face indices, shape [3, num_faces].

    Returns:
    tuple: normals [num_faces, 3], corners [num_faces, 9], centers [num_faces, 3]
    """
    v0 = pos[face[0]]  # [num_faces, 3]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    # Compute face normals
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = F.normalize(normals, p=2, dim=1)  # [num_faces, 3]

    # Compute face centers
    centers = (v0 + v1 + v2) / 3.0  # [num_faces, 3]

    # Compute corner features (flattened vertex positions of each face)
    corners = torch.cat([v0, v1, v2], dim=1)  # [num_faces, 9]

    return normals, corners, centers

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
        mri_folder = os.path.join(subject_folder, 'mri')
        if os.path.exists(surf_folder) and os.path.exists(label_folder) and os.path.exists(mri_folder):
            data.append((surf_folder, label_folder, mri_folder))
            labels.append(1)  # Label 1 for MND patients

    # Load Normal Controls
    for i in os.listdir(normal_folder):
        subject_folder = os.path.join(normal_folder, i)
        surf_folder = os.path.join(subject_folder, 'surf')
        label_folder = os.path.join(subject_folder, 'label')
        mri_folder = os.path.join(subject_folder, 'mri')
        if os.path.exists(surf_folder) and os.path.exists(label_folder) and os.path.exists(mri_folder):
            data.append((surf_folder, label_folder, mri_folder))
            labels.append(0)  # Label 0 for Normal Controls

    return data, labels

# Function to create DataLoader from the dataset
def create_dataloader(
    data,
    labels,
    batch_size=32,
    augment=False,
    target_faces=1024,
    ba_labels_channel_1=['BA4a', 'BA4p'],
    ba_labels_channel_2=['BA6'],
    cc_label_ids=[251, 253, 255],  # Corpus callosum labels in ASEG
):
    """
    Create a DataLoader for batched mesh data with multiple regions.

    Parameters:
    data (list): List of tuples containing surf_folder, label_folder, and mri_folder paths.
    labels (list): List of labels corresponding to the data.
    batch_size (int): Number of samples per batch.
    augment (bool): Whether to apply data augmentation.
    target_faces (int): Target number of faces per region after simplification.
    ba_labels_channel_1 (list): List of BA labels for channel 1.
    ba_labels_channel_2 (list): List of BA labels for channel 2.
    cc_label_ids (list): List of label IDs for the corpus callosum regions in ASEG.

    Returns:
    DataLoader: PyTorch Geometric DataLoader object.
    """
    dataset = []
    for (surf_folder, label_folder, mri_folder), label in zip(data, labels):
        # Initialize lists for normals, corners, centers per region
        region_normals = []
        region_corners = []
        region_centers = []

        # Process BA regions (channels 1 and 2)
        hemispheres = ['lh', 'rh']
        for ba_labels in [ba_labels_channel_1, ba_labels_channel_2]:
            all_normals = []
            all_corners = []
            all_centers = []
            for hemi in hemispheres:
                pial_file = os.path.join(surf_folder, f'{hemi}.pial')
                # List of label files for the BA regions
                label_files = [os.path.join(label_folder, f'{hemi}.{label}_exvivo.label') for label in ba_labels]
                if os.path.exists(pial_file):
                    vertices, faces = extract_regions_from_pial_using_labels(pial_file, label_files)
                    if vertices is None or faces is None:
                        continue  # Skip if no region found
                    # Simplify the mesh
                    vertices, faces = simplify_mesh(vertices, faces, target_faces=target_faces)
                    # If augment is True, apply augmentation to vertices
                    if augment:
                        vertices = augment_vertices(vertices)
                    # Convert to tensors
                    pos = torch.tensor(vertices, dtype=torch.float)
                    face = torch.tensor(faces.T, dtype=torch.long)  # [3, num_faces]
                    # Compute normals, corners, centers
                    normals, corners, centers = compute_face_features(pos, face)
                    all_normals.append(normals)
                    all_corners.append(corners)
                    all_centers.append(centers)
                else:
                    print(f'Missing pial file for {surf_folder}')
            if len(all_normals) == 0:
                continue  # Skip this subject if no data for this channel
            # Concatenate data from all hemispheres
            normals = torch.cat(all_normals, dim=0)
            corners = torch.cat(all_corners, dim=0)
            centers = torch.cat(all_centers, dim=0)
            # Enforce uniform number of faces per region
            desired_faces = target_faces * len(hemispheres)
            current_faces = normals.size(0)
            if current_faces < desired_faces:
                pad_size = desired_faces - current_faces
                normals = torch.cat([normals, normals[-1].unsqueeze(0).repeat(pad_size, 1)], dim=0)
                corners = torch.cat([corners, corners[-1].unsqueeze(0).repeat(pad_size, 1)], dim=0)
                centers = torch.cat([centers, centers[-1].unsqueeze(0).repeat(pad_size, 1)], dim=0)
            elif current_faces > desired_faces:
                normals = normals[:desired_faces]
                corners = corners[:desired_faces]
                centers = centers[:desired_faces]
            region_normals.append(normals)
            region_corners.append(corners)
            region_centers.append(centers)

        # Process Corpus Callosum (channel 3)
        aseg_file = os.path.join(mri_folder, 'aseg.mgz')
        if os.path.exists(aseg_file):
            vertices, faces = extract_cc_from_aseg(aseg_file, cc_label_ids, apply_affine=True)
            if vertices is None or faces is None:
                continue  # Skip if no data
            # Simplify the mesh
            vertices, faces = simplify_mesh(vertices, faces, target_faces=target_faces)
            # If augment is True, apply augmentation to vertices
            if augment:
                vertices = augment_vertices(vertices)
            # Convert to tensors
            pos = torch.tensor(vertices, dtype=torch.float)
            face = torch.tensor(faces.T, dtype=torch.long)  # [3, num_faces]
            # Compute normals, corners, centers
            normals, corners, centers = compute_face_features(pos, face)
            # Enforce uniform number of faces per region
            desired_faces = target_faces  # Since only one region here
            current_faces = normals.size(0)
            if current_faces < desired_faces:
                pad_size = desired_faces - current_faces
                normals = torch.cat([normals, normals[-1].unsqueeze(0).repeat(pad_size, 1)], dim=0)
                corners = torch.cat([corners, corners[-1].unsqueeze(0).repeat(pad_size, 1)], dim=0)
                centers = torch.cat([centers, centers[-1].unsqueeze(0).repeat(pad_size, 1)], dim=0)
            elif current_faces > desired_faces:
                normals = normals[:desired_faces]
                corners = corners[:desired_faces]
                centers = centers[:desired_faces]
            region_normals.append(normals)
            region_corners.append(corners)
            region_centers.append(centers)
        else:
            print(f'Missing ASEG file for {mri_folder}')
            continue  # Skip this subject if ASEG file is missing

        # Check if data is available for all regions
        if len(region_normals) != 3:
            continue  # Ensure all channels are present

        # Create Data object
        data_obj = Data(
            normals=region_normals,   # List of tensors per region
            corners=region_corners,
            centers=region_centers,
            y=torch.tensor([label], dtype=torch.long)
        )
        dataset.append(data_obj)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Structural Descriptor: Face Kernel Correlation and Face Rotate Convolution
class StructuralDescriptor(nn.Module):
    def __init__(self):
        super(StructuralDescriptor, self).__init__()
        self.fc_correlation = nn.Conv1d(3, 64, 1)  # Face Kernel Correlation (normal vectors)
        self.fc_rotation = nn.Conv1d(9, 64, 1)     # Face Rotate Convolution (corner features)

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

# MeshNet model with separate convolutional paths for each region and dropout
class MeshNetSeparatePaths(nn.Module):
    def __init__(self, num_classes=2, num_regions=3, dropout_rate=0.3):
        super(MeshNetSeparatePaths, self).__init__()
        self.num_regions = num_regions
        self.dropout_rate = dropout_rate

        # Define descriptors and convolutions for each region
        self.structural_descriptors = nn.ModuleList([StructuralDescriptor() for _ in range(num_regions)])
        self.spatial_descriptors = nn.ModuleList([SpatialDescriptor() for _ in range(num_regions)])

        self.conv1 = nn.ModuleList([nn.Conv1d(192, 256, 1) for _ in range(num_regions)])
        self.conv2 = nn.ModuleList([nn.Conv1d(256, 512, 1) for _ in range(num_regions)])
        self.conv3 = nn.ModuleList([nn.Conv1d(512, 512, 1) for _ in range(num_regions)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(num_regions)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(512) for _ in range(num_regions)])
        self.bn3 = nn.ModuleList([nn.BatchNorm1d(512) for _ in range(num_regions)])
        self.dropout = nn.Dropout(dropout_rate)

        # Global pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers (MLP)
        self.fc1 = nn.Linear(512 * num_regions, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, data):
        batch_size = data.y.size(0)
        region_features = []

        for i in range(self.num_regions):
            normals = data.normals[i]
            corners = data.corners[i]
            centers = data.centers[i]

            # Reshape and transpose to [batch_size, num_features, num_faces]
            num_faces = normals.size(0) // batch_size
            normal = normals.view(batch_size, num_faces, -1).transpose(1, 2)
            corner = corners.view(batch_size, num_faces, -1).transpose(1, 2)
            center = centers.view(batch_size, num_faces, -1).transpose(1, 2)

            # Structural and Spatial descriptors
            struct_features = self.structural_descriptors[i](normal, corner)  # [batch_size, 128, num_faces]
            spatial_features = self.spatial_descriptors[i](center)  # [batch_size, 64, num_faces]

            combined_features = torch.cat([struct_features, spatial_features], dim=1)  # [batch_size, 192, num_faces]

            x = F.relu(self.bn1[i](self.conv1[i](combined_features)))
            x = F.relu(self.bn2[i](self.conv2[i](x)))
            x = F.relu(self.bn3[i](self.conv3[i](x)))
            x = self.pool(x).squeeze(-1)  # [batch_size, 512]
            region_features.append(x)

        # Concatenate features from all regions
        x = torch.cat(region_features, dim=1)  # [batch_size, 512 * num_regions]

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

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

    # Loss
    axs[0, 0].plot(epochs, metrics_dict['train_loss'], label='Train Loss', alpha=0.3)
    axs[0, 0].plot(epochs, exponential_smoothing(metrics_dict['train_loss']), label='Smoothed Train Loss')
    axs[0, 0].plot(epochs, metrics_dict['val_loss'], label='Val Loss', alpha=0.3)
    axs[0, 0].plot(epochs, exponential_smoothing(metrics_dict['val_loss']), label='Smoothed Val Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()

    # Accuracy
    axs[0, 1].plot(epochs, metrics_dict['val_acc'], label='Val Accuracy', alpha=0.3)
    axs[0, 1].plot(epochs, exponential_smoothing(metrics_dict['val_acc']), label='Smoothed Val Accuracy')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].legend()

    # Precision
    axs[1, 0].plot(epochs, metrics_dict['val_prec'], label='Val Precision', alpha=0.3)
    axs[1, 0].plot(epochs, exponential_smoothing(metrics_dict['val_prec']), label='Smoothed Val Precision')
    axs[1, 0].set_title('Validation Precision')
    axs[1, 0].legend()

    # Recall
    axs[1, 1].plot(epochs, metrics_dict['val_rec'], label='Val Recall', alpha=0.3)
    axs[1, 1].plot(epochs, exponential_smoothing(metrics_dict['val_rec']), label='Smoothed Val Recall')
    axs[1, 1].set_title('Validation Recall')
    axs[1, 1].legend()

    fig.suptitle(f"Training Curves (lr={hyperparams[0]}, dropout={hyperparams[1]}, batch_size={hyperparams[2]})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"./models/meshnet_v2_mc_BA/training_curves/training_curves_lr{hyperparams[0]}_dropout{hyperparams[1]}_batch_size_{hyperparams[2]}.png")
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(conf_mt_counter, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'./models/meshnet_v2_mc_BA/confusion_matrix/confusion_matrix_{conf_mt_counter}.png')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    mnd_folder = './MND_patients'
    normal_folder = './Normal_control'

    # Create directories for saving models and figures
    os.makedirs('./models/meshnet_v2_mc_BA/confusion_matrix', exist_ok=True)
    os.makedirs('./models/meshnet_v2_mc_BA/hyperparameters', exist_ok=True)
    os.makedirs('./models/meshnet_v2_mc_BA/training_curves', exist_ok=True)

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
    learning_rates = [ 0.0001,0.00001,0.000001]
    dropout_rates = [0, 0.3, 0.5]
    batch_sizes = [64, 128]
    target_faces = 1024  # Adjusted target_faces per region to enforce uniformity

    best_val_acc = 0
    best_hyperparams = None
    results = []
    conf_mt_counter = 0
    all_metrics = []

    total_combinations = len(learning_rates) * len(dropout_rates) * len(batch_sizes)
    print(f"Total hyperparameter combinations: {total_combinations}")

    for lr, dropout, batch_size in product(learning_rates, dropout_rates, batch_sizes):
        print(f"\nTraining with lr={lr}, dropout={dropout}, batch_size={batch_size}, augment=False")

        # Create DataLoaders with multiple regions
        train_loader = create_dataloader(
            train_data, train_labels, batch_size=batch_size, augment=False,
            target_faces=target_faces,
            ba_labels_channel_1=['BA4a', 'BA4p'],
            ba_labels_channel_2=['BA6'],
            cc_label_ids=[251, 253, 255],  # Corpus callosum labels
        )
        val_loader = create_dataloader(
            val_data, val_labels, batch_size=batch_size, augment=False,
            target_faces=target_faces,
            ba_labels_channel_1=['BA4a', 'BA4p'],
            ba_labels_channel_2=['BA6'],
            cc_label_ids=[251, 253, 255],  # Corpus callosum labels
        )

        # Initialize model, optimizer, and learning rate scheduler
        model = MeshNetSeparatePaths(num_classes=2, num_regions=3, dropout_rate=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Training loop
        metrics_dict = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': []}
        for epoch in tqdm(range(1, 101), desc=f"LR={lr}, Dropout={dropout}, batch_size={batch_size}"):
            train_loss = train(model, train_loader, optimizer, device, class_weights)
            val_loss, val_acc, val_prec, val_rec, val_preds, val_labels_epoch = validate(model, val_loader, device, class_weights)
            scheduler.step(val_loss)
            metrics_dict['train_loss'].append(train_loss)
            metrics_dict['val_loss'].append(val_loss)
            metrics_dict['val_acc'].append(val_acc)
            metrics_dict['val_prec'].append(val_prec)
            metrics_dict['val_rec'].append(val_rec)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}')

        # Plot training curves
        plot_training_curves(metrics_dict, hyperparams=(lr, dropout, batch_size))

        # Plot confusion matrix after last epoch
        plot_confusion_matrix(conf_mt_counter, val_labels_epoch, val_preds, classes=['Normal', 'MND'])
        conf_mt_counter += 1

        # Record results
        results.append((lr, dropout, batch_size, metrics_dict['val_acc'][-1]))
        all_metrics.append({
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size,
            'metrics': metrics_dict
        })

        # Save best model
        if metrics_dict['val_acc'][-1] > best_val_acc:
            best_val_acc = metrics_dict['val_acc'][-1]
            best_hyperparams = (lr, dropout, batch_size)
            torch.save(model.state_dict(), './models/meshnet_v2_mc_BA/best_model.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # Plot hyperparameter search results
    plt.figure(figsize=(10, 7))
    for lr in learning_rates:
        for dropout in dropout_rates:
            lr_dropout_results = [r for r in results if r[0] == lr and r[1] == dropout]
            x = [r[2] for r in lr_dropout_results]  # batch sizes
            y = [r[3] for r in lr_dropout_results]  # validation accuracies
            plt.plot(x, y, 'o-', label=f'lr={lr}, dropout={dropout}')
            # Apply exponential smoothing
            y_smoothed = exponential_smoothing(y, alpha=0.9)
            plt.plot(x, y_smoothed, '-', label=f'lr={lr}, dropout={dropout} Smoothed')
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Hyperparameter Search Results')
    plt.legend()
    plt.savefig('./models/meshnet_v2_mc_BA/hyperparameters/search_results.png')
    plt.show()

    print(f"Best hyperparameters: lr={best_hyperparams[0]}, dropout={best_hyperparams[1]}, batch_size={best_hyperparams[2]}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save all metrics for analysis
    torch.save(all_metrics, './models/meshnet_v2_mc_BA/hyperparameters/all_metrics.pt')

    print("Training and testing completed.")

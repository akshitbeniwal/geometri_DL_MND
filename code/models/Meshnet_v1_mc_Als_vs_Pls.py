import os
import numpy as np
import pandas as pd  # For handling CSV
import torch
from torch_sparse import SparseTensor
import torch.nn as nn
import torch.nn.functional as F
import nibabel.freesurfer.io as fsio
from torch_geometric.data import Data, DataLoader as GeometricDataLoader  # Alias DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import trimesh
from torch_scatter import scatter_mean  # Ensure torch_scatter is installed
import warnings
from torch.utils.data import WeightedRandomSampler  # Removed DataLoader import to prevent conflict

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print("MeshNet ALS vs PLS Started")
print(f"Trimesh version: {trimesh.__version__}")

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`simplify_quadratic_decimation` is deprecated as it was a typo and will be removed in March 2024: replace with `simplify_quadric_decimation`")
warnings.filterwarnings("ignore", category=UserWarning, message="The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.")

# Function to simplify mesh using Trimesh
def simplify_mesh(vertices, faces, target_faces=1024):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) > target_faces:
        simplified_mesh = mesh.simplify_quadric_decimation(target_faces)  # Updated method name
        return simplified_mesh.vertices, simplified_mesh.faces
    return vertices, faces

# Function to extract multiple regions from .pial and .annot files
def extract_regions_from_pial(pial_file, annot_file, target_region_names):
    """
    Extract the surface meshes of multiple regions from the .pial file based on the annotation.

    Parameters:
    pial_file (str): Path to the .pial file.
    annot_file (str): Path to the .annot file.
    target_region_names (list): List of target region names (e.g., ['G_precentral', 'G_paracentral']).

    Returns:
    list of tuples: Each tuple contains extracted vertices and faces for a target region.
    """
    # Load the pial file
    vertices, faces = fsio.read_geometry(pial_file)
    faces = faces.astype(np.int64)

    # Load the annotation file
    labels, ctab, names = fsio.read_annot(annot_file)
    # Decode the region names
    names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in names]

    extracted_regions = []

    for target_region_name in target_region_names:
        if target_region_name in names:
            target_label_index = names.index(target_region_name)
            target_label = ctab[target_label_index, -1]  # The label ID
        else:
            print(f"Region {target_region_name} not found in annotation file")
            extracted_regions.append((None, None))
            continue

        # Create mask for vertices belonging to the target region
        region_mask = labels == target_label_index

        # Extract the indices of vertices that belong to the target region
        region_indices = np.where(region_mask)[0]

        if len(region_indices) == 0:
            print(f"No vertices found for region {target_region_name} in {pial_file}")
            extracted_regions.append((None, None))
            continue

        # Create a map from original indices to new indices for the target region
        index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(region_indices)}

        # Extract the vertices for the target region
        region_vertices = vertices[region_indices]

        # Find faces that are composed of vertices all within the region
        valid_faces = []
        for face in faces:
            if all(vertex in index_map for vertex in face):
                mapped_face = [index_map[vertex] for vertex in face]
                valid_faces.append(mapped_face)

        if len(valid_faces) == 0:
            print(f"No faces found for region {target_region_name} in {pial_file}")
            extracted_regions.append((None, None))
            continue

        region_faces = np.array(valid_faces)

        extracted_regions.append((region_vertices, region_faces))

    return extracted_regions  # List of (vertices, faces) tuples

# Function for data augmentation
def augment_vertices(vertices):
    # Random Rotation
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
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
    tuple: face_features [num_faces, 8], edge_index [2, num_edges].
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

    # Stack spatial features
    face_features = torch.cat([normals, centers, areas], dim=1)  # Shape: [num_faces, 7]

    # Build edge indices for adjacency
    num_faces = face.shape[1]
    face_np = face.numpy().T  # [num_faces, 3]

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
    face_features = torch.cat([face_features, struct_features], dim=1)  # Shape: [num_faces, 8]

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

# Function to load data with labels from CSV
def load_data(mnd_folder, csv_path):
    """
    Load data from the MND_patients directory and assign labels based on the CSV file.

    Parameters:
    - mnd_folder (str): Path to the MND_patients directory containing subject folders.
    - csv_path (str): Path to the 'New_MND_patient_data_summary.csv' file.

    Returns:
    - data (list of tuples): Each tuple contains (surf_folder_path, annot_folder_path).
    - labels (list of int): Corresponding labels for each data sample (0 for ALS, 1 for PLS).
    """
    data = []
    labels = []

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The CSV file at {csv_path} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The CSV file at {csv_path} is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV file: {e}")

    # Ensure required columns exist
    required_columns = {'folder', 'label'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"The CSV file is missing required columns: {missing}")

    # Filter the dataframe to include only ALS (1) and PLS (2) labels
    df_filtered = df[df['label'].isin([1, 2])].copy()

    # Convert 'folder' column to string to match folder names in the directory
    df_filtered['folder'] = df_filtered['folder'].astype(str)

    # Create a mapping from folder name to label
    folder_to_label = dict(zip(df_filtered['folder'], df_filtered['label']))

    # Iterate through each folder in the MND_patients directory
    for folder_name in os.listdir(mnd_folder):
        folder_path = os.path.join(mnd_folder, folder_name)

        # Check if the current path is a directory
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_path} as it is not a directory.")
            continue

        # Check if the folder name exists in the CSV mapping
        if folder_name in folder_to_label:
            label = folder_to_label[folder_name]

            # Map label 1 (ALS) to 0 and label 2 (PLS) to 1 for PyTorch compatibility
            label_mapped = label - 1  # 1 -> 0, 2 -> 1

            # Define paths to 'surf' and 'label' subfolders
            surf_folder = os.path.join(folder_path, 'surf')
            annot_folder = os.path.join(folder_path, 'label')

            # Verify that both 'surf' and 'label' subfolders exist
            if os.path.exists(surf_folder) and os.path.exists(annot_folder):
                data.append((surf_folder, annot_folder))
                labels.append(label_mapped)
                print(f"Loaded subject {folder_name} with label {'ALS' if label_mapped == 0 else 'PLS'}.")
            else:
                print(f"Missing 'surf' or 'label' folder for subject {folder_name}. Skipping.")
        else:
            print(f"Folder {folder_name} not found in CSV labels. Skipping.")

    print(f"Total samples loaded: {len(data)}")
    print(f"Labels distribution: {pd.Series(labels).value_counts().to_dict()}")

    return data, labels

def create_dataloader(data, labels, batch_size=32, augment=False, target_faces=1024, target_region_names=['precentral', 'paracentral'], is_train=True):
    """
    Create a DataLoader for batched mesh data with multiple regions.

    Parameters:
    - data (list): List of tuples containing surf_folder and annot_folder paths.
    - labels (list): List of labels corresponding to the data.
    - batch_size (int): Number of samples per batch.
    - augment (bool): Whether to apply data augmentation.
    - target_faces (int): Target number of faces per region after simplification.
    - target_region_names (list): List of region names to extract.
    - is_train (bool): Flag indicating whether the DataLoader is for training or validation.

    Returns:
    - DataLoader: PyTorch Geometric DataLoader object.
    """
    dataset = []
    for (surf_folder, annot_folder), label in zip(data, labels):
        # Process both hemispheres
        hemispheres = ['lh', 'rh']
        all_vertices = []
        all_faces = []
        offset = 0  # To ensure face indices are correct when combining hemispheres

        for hemi in hemispheres:
            pial_file = os.path.join(surf_folder, f'{hemi}.pial')
            annot_file = os.path.join(annot_folder, f'{hemi}.aparc.DKTatlas.annot')
            if os.path.exists(pial_file) and os.path.exists(annot_file):
                extracted_regions = extract_regions_from_pial(pial_file, annot_file, target_region_names)
                for vertices, faces in extracted_regions:
                    if vertices is None or faces is None:
                        continue  # Skip if no region found
                    # Simplify the mesh
                    vertices, faces = simplify_mesh(vertices, faces, target_faces=target_faces)
                    # If augment is True, apply augmentation to vertices
                    if augment:
                        vertices = augment_vertices(vertices)
                    # Adjust face indices when combining hemispheres and regions
                    faces = faces + offset
                    offset += len(vertices)
                    all_vertices.append(vertices)
                    all_faces.append(faces)
            else:
                print(f'Missing pial or annot file for {surf_folder}')

        if len(all_vertices) == 0:
            continue  # Skip this subject if no data

        # Combine vertices and faces from all hemispheres and regions
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)

        # Enforce uniform number of faces per graph
        desired_faces = target_faces * len(target_region_names) * 2  # regions * hemispheres
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
            y=y
        )
        dataset.append(data_obj)

    if not dataset:
        raise ValueError("No data found. Please check your data directories and CSV file.")

    # Separate labels for sampler if training
    if is_train:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        class_sample_count = np.array([len(np.where(labels_tensor.numpy() == t)[0]) for t in np.unique(labels_tensor.numpy())])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels_tensor.numpy()])
        sampler = WeightedRandomSampler(weights=torch.from_numpy(samples_weight).float(), num_samples=len(samples_weight), replacement=True)
        shuffle = False  # When using sampler, shuffle must be False
    else:
        sampler = None
        shuffle = False  # Typically, shuffle=False for validation/testing

    return GeometricDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)  # Use GeometricDataLoader

# Define the EnhancedMeshNet model with Separate Convolutional Paths for Each Region
class EnhancedMeshNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3, num_regions=2):
        super(EnhancedMeshNet, self).__init__()
        self.num_regions = num_regions
        # Each region has 8 features
        self.conv1 = nn.ModuleList([nn.Conv1d(8, 64, kernel_size=1) for _ in range(num_regions)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(64) for _ in range(num_regions)])
        self.conv2 = nn.ModuleList([nn.Conv1d(64, 128, kernel_size=1) for _ in range(num_regions)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(num_regions)])
        self.conv3 = nn.ModuleList([nn.Conv1d(128, 256, kernel_size=1) for _ in range(num_regions)])
        self.bn3 = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(num_regions)])

        # Global pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * num_regions, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        # Handle batched data
        batch_size = data.num_graphs
        num_faces = data.face.size(-1) // (self.num_regions * batch_size)  # Updated from data.num_faces

        region_features = []
        for region_idx in range(self.num_regions):
            # Extract features for each region
            x = data.face_features.view(batch_size, self.num_regions, 8, num_faces)[:, region_idx, :, :]  # [batch_size, 8, num_faces]
            x = F.relu(self.bn1[region_idx](self.conv1[region_idx](x)))  # [batch_size, 64, num_faces]
            x = F.relu(self.bn2[region_idx](self.conv2[region_idx](x)))  # [batch_size, 128, num_faces]
            x = F.relu(self.bn3[region_idx](self.conv3[region_idx](x)))  # [batch_size, 256, num_faces]
            x = self.pool(x)  # [batch_size, 256, 1]
            x = x.view(x.size(0), -1)  # [batch_size, 256]
            region_features.append(x)

        # Concatenate features from all regions
        x = torch.cat(region_features, dim=1)  # [batch_size, 256 * num_regions]

        x = F.relu(self.bn4(self.fc1(x)))  # [batch_size, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, num_classes]

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
    plt.savefig(f"./models/meshnet_mc_ALS_vs_PLS_2/training_curves/training_curves_lr{hyperparams[0]}_dropout{hyperparams[1]}_batch_size_{hyperparams[2]}.png")
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
    plt.savefig(f'./models/meshnet_mc_ALS_vs_PLS_2/confusion_matrix/confusion_matrix_{conf_mt_counter}.png')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    mnd_folder = './MND_patients'
    csv_path = './New_MND_patient_data_summary.csv'  # Path to your CSV file

    # Create directories for saving models and figures
    os.makedirs('./models/meshnet_mc_ALS_vs_PLS_2/confusion_matrix', exist_ok=True)
    os.makedirs('./models/meshnet_mc_ALS_vs_PLS_2/hyperparameters', exist_ok=True)
    os.makedirs('./models/meshnet_mc_ALS_vs_PLS_2/training_curves', exist_ok=True)

    # Load data
    print("Loading data...")
    data, labels = load_data(mnd_folder, csv_path)
    print(f"Total samples loaded: {len(data)}")

    if len(data) == 0:
        raise ValueError("No data loaded. Please check the CSV file and folder structure.")

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
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0,0.3, 0.5]
    batch_sizes = [16, 32, 64,128]  # Adjust based on your dataset size
    target_faces = 2058  # Adjusted target_faces per region to enforce uniformity

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
            target_faces=target_faces, target_region_names=['precentral', 'paracentral'], is_train=True
        )
        val_loader = create_dataloader(
            val_data, val_labels, batch_size=batch_size, augment=False,
            target_faces=target_faces, target_region_names=['precentral', 'paracentral'], is_train=False
        )

        # Initialize model, optimizer, and learning rate scheduler
        model = EnhancedMeshNet(num_classes=2, dropout_rate=dropout, num_regions=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # Training loop
        metrics_dict = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': []}
        for epoch in tqdm(range(1, 51), desc=f"LR={lr}, Dropout={dropout}, batch_size={batch_size}"):
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
        plot_confusion_matrix(conf_mt_counter, val_labels_epoch, val_preds, classes=['ALS', 'PLS'])
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
            torch.save(model.state_dict(), './models/meshnet_mc_ALS_vs_PLS_2/best_model.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # Plot hyperparameter search results
    plt.figure(figsize=(10, 7))
    combination_indices = range(1, len(results) + 1)
    for lr in learning_rates:
        lr_results = [r for r in results if r[0] == lr]
        x = [i+1 for i in range(len(lr_results))]
        y = [r[3] for r in lr_results]
        plt.plot(x, y, 'o-', label=f'lr={lr}')
        # Apply exponential smoothing
        y_smoothed = exponential_smoothing(y, alpha=0.9)
        plt.plot(x, y_smoothed, '-', label=f'lr={lr} Smoothed')
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Validation Accuracy')
    plt.title('Hyperparameter Search Results')
    plt.legend()
    plt.savefig('./models/meshnet_mc_ALS_vs_PLS_2/hyperparameters/search_results.png')
    plt.show()

    print(f"Best hyperparameters: lr={best_hyperparams[0]}, dropout={best_hyperparams[1]}, batch_size={best_hyperparams[2]}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save all metrics for analysis
    torch.save(all_metrics, './models/meshnet_mc_ALS_vs_PLS_2/hyperparameters/all_metrics.pt')

    print("Training and testing completed.")

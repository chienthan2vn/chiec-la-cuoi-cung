import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import time
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_recall_fscore_support
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir: str, batch_size: int, image_size: int, max_samples: int = 1000):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    subset_datasets = {}
    for x in ['train', 'test']:
        full_ds = image_datasets[x]
        num_samples = min(len(full_ds), max_samples)
        indices = torch.randperm(len(full_ds))[:num_samples]
        subset_datasets[x] = Subset(full_ds, indices)
    
    dataloaders = {x: DataLoader(subset_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=0)
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(subset_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=5):
    best_acc = 0.0
    for epoch in range(num_epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch}", leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )

            # Log to MLflow
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_acc", float(epoch_acc), step=epoch)
            mlflow.log_metric(f"{phase}_precision", float(precision), step=epoch)
            mlflow.log_metric(f"{phase}_recall", float(recall), step=epoch)
            mlflow.log_metric(f"{phase}_f1", float(f1), step=epoch)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Log model weights and register it to MLflow Model Registry
                mlflow.pytorch.log_model(
                    model, 
                    "model",
                    registered_model_name=os.getenv("MLFLOW_MODEL_NAME")
                )

    return float(best_acc)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(SEED)
    
    # 1. Load Data
    data_dir = os.path.join(hydra.utils.get_original_cwd(), "data/raw/PlantDoc-Dataset")
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir, 
        cfg.train.batch_size, 
        cfg.train.image_size, 
        max_samples=cfg.train.max_samples
    )

    # 2. MLflow Experiment Setup
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(cfg.experiment_name)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"trial_{cfg.optimizer.lr:.4f}_{cfg.train.max_samples}_{timestamp}"
    
    # We use a single run, Hydra Multirun will create multiple runs
    with mlflow.start_run(run_name=run_name):
        # Log config as params
        params = OmegaConf.to_container(cfg, resolve=True)
        # Flatten params for mlflow if needed, but nested is okay for some UI
        mlflow.log_params(params["train"])
        mlflow.log_params(params["optimizer"])
        mlflow.log_params(params["model"])

        # 3. Build & Train
        model = build_model(len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
        
        accuracy = train_model(
            model, 
            dataloaders, 
            dataset_sizes, 
            criterion, 
            optimizer, 
            num_epochs=cfg.train.num_epochs
        )
        
        print(f"Final Test Accuracy: {accuracy}")
        
        # 4. Log Hydra logs as artifacts
        # mlflow.log_artifacts(".")
            
        return accuracy

if __name__ == "__main__":
    main()

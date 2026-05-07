import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time

# --- CONFIGURATION ---
DATA_DIR = "data/raw/PlantDoc-Dataset"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir: str, batch_size: int, image_size: int):
    # Augmentation cho tập train, chỉ normalize cho tập test
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
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def build_model(num_classes: int):
    # Load ResNet50 pre-trained
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze initial layers (optional, but good for Phase 0 R&D)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Thay đổi lớp cuối cùng cho phù hợp với số lượng lớp của PlantDoc
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(DEVICE)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10):
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Lưu weights sau khi train
    torch.save(model.state_dict(), 'resnet50_plantdoc_phase0.pth')
    print("Model saved as resnet50_plantdoc_phase0.pth")

if __name__ == "__main__":
    set_seed(SEED)
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    print(f"Classes: {class_names}")
    print(f"Train size: {dataset_sizes['train']}, Test size: {dataset_sizes['test']}")
    
    # 2. Build Model
    model = build_model(len(class_names))
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Train
    train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=NUM_EPOCHS)

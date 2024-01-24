import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
class CalisthenicsNet(nn.Module):
    def __init__(self):
        super(CalisthenicsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
class CalisthenicsDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform

        self.image_files = []
        self.labels = []

        correct_folder = os.path.join(root_folder, 'correct')
        for image_file in os.listdir(correct_folder):
            self.image_files.append(os.path.join(correct_folder, image_file))
            self.labels.append(1)

  
        incorrect_folder = os.path.join(root_folder, 'incorrect')
        for image_file in os.listdir(incorrect_folder):
            self.image_files.append(os.path.join(incorrect_folder, image_file))
            self.labels.append(0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


root_folder = "handstand_images"
calisthenics_dataset = CalisthenicsDataset(root_folder, transform)

dataset_size = len(calisthenics_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(calisthenics_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = CalisthenicsNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 5
best_val_loss = float('inf')
patience = 3
learning_rate = 0.0001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0

    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        batch_data, batch_labels = batch_data.to(device), batch_labels.float().to(device)
        outputs = model(batch_data)
        loss = criterion(outputs.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted_train = (outputs > 0.5).float()
        correct_predictions_train += (predicted_train == batch_labels.unsqueeze(1).to(device)).sum().item()
        total_samples_train += batch_labels.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy_train = (correct_predictions_train / total_samples_train) * 100 if total_samples_train != 0 else 0
    print(f'Training - Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}, Accuracy: {accuracy_train:.2f}%')

 
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct_predictions_val = 0
        total_samples_val = 0

        for val_data, val_labels in val_loader:
            val_data, val_labels = val_data.to(device), val_labels.float().to(device)
            val_outputs = model(val_data)
            val_loss += criterion(val_outputs.squeeze(), val_labels).item()

 
            predicted_val = (val_outputs > 0.5).float()
            correct_predictions_val += (predicted_val == val_labels.unsqueeze(1).to(device)).sum().item()
            total_samples_val += val_labels.size(0)

        average_val_loss = val_loss / len(val_loader)
        accuracy_val = (correct_predictions_val / total_samples_val) * 100 if total_samples_val != 0 else 0
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {accuracy_val:.2f}%')

   
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
        else:
            print(f'Validation loss increased. Early stopping after {patience} epochs.')
            break

    scheduler.step() 


torch.save(model.state_dict(), 'calisthenics_hs.pth')
print('Model weights saved successfully!')

import os
import time
import copy
from typing import Tuple
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from googlenet_pytorch import googlenet


# Default setting
BATCH_SIZE = 100
EPOCHS = 80
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current Device:', DEVICE)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(DATA_ROOT, os.pardir))
DATA_ROOT = os.path.join(DATA_ROOT, 'data')

train_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT,
                                             train=True,
                                             transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT,
                                            train=False,
                                            transform=transforms.ToTensor())
# DataLoader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)
print("Train data:", len(train_loader) * 100)
print("Test data:", len(test_loader) * 100)

model = googlenet(pretrained=True, aux_logits=True)
model.aux1.fc2 = nn.Linear(1024, 10)
model.aux2.fc2 = nn.Linear(1024, 10)
model.fc = nn.Linear(1024, 10)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_model(model, criterion, dataloaders, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 에폭(epoch)은 학습과 검증 단계를 가짐
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)  

                # 매개변수 경사도 0으로 설정
                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # outputs이 namedTuple일 경우 보조 분류기에 대한 처리 필요
                    if isinstance(outputs, Tuple):
                        outputs = outputs[0]
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders)
            epoch_acc = running_corrects.double() / len(dataloaders)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 모델을 깊은 복사
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model
    

if __name__ == "__main__":
    model_ft = train_model(model, criterion, train_loader, optimizer, 
                           exp_lr_scheduler, num_epochs=25)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
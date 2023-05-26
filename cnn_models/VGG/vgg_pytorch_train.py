import time
import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt

from vgg_pytorch import vgg16


plt.ion()   # interactive mode

batch_size = 128
num_epochs = 3
learning_rate = 1e-3

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

dataloaders = dict()
dataloaders['train'] = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
dataloaders['val'] = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

print("train 개수:", dataset_sizes['train'])
print("test 개수:", dataset_sizes['val'])

class_names = train_set.classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('class names:', class_names)
print(device)

# show data
def imshow(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_data():
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    inputs_ = inputs[:3]
    classes_ = classes[:3]

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs_)

    imshow(out, title=[class_names[x] for x in classes_])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            iteration_count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                iteration_count += len(inputs)
                print(f"Iteration {iteration_count} / {dataset_sizes[phase]}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in traininig phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_loss_list.append(epoch_loss)
            else:
                val_acc_list.append(epoch_acc)
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_list, val_acc_list


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":
    model_ft = vgg16(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    print(model_ft)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, train_loss_list, val_acc_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
    

    #plot train loss 
    # x=[i for i in range(0,num_epochs)]
    # plt.title("Train Loss")
    # plt.xticks(x)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.plot(x,train_loss_list)
    # plt.show()

    #plot test acc
    # x=[i for i in range(0,num_epochs)]
    # plt.title("Test Accuracy")
    # plt.xticks(x)
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.plot(x,val_acc_list)
    # plt.show()

    visualize_model(model_ft)

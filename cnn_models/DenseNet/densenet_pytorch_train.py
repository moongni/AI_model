from __future__ import print_function, division
import os
import time
import copy

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from densenet_pytorch import densenet161


#Batch size
batch_size=16
#Epoch 
num_epochs=3
#learning rate
learning_rate=0.001

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(DATA_ROOT, os.pardir))
DATA_ROOT = os.path.join(DATA_ROOT, 'data')

train_set=torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=data_transforms['train'])
test_set =torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=data_transforms['val'])

dataloaders=dict()
dataloaders['train']= torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
dataloaders['val']= torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

print("train 개수",dataset_sizes['train'])
print("test 개수",dataset_sizes['val'])

class_names = train_set.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("class_names:",class_names)
print(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_list=[]
    val_acc_list=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            iteration_count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                iteration_count += len(inputs)
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

                    # backward + optimize only if in training phase
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase=="train":
              train_loss_list.append(epoch_loss)
            elif phase=="val":
              val_acc_list.append(epoch_acc)


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_loss_list,val_acc_list

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

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":
    model_ft = densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier= nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    print(model_ft)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #SGD
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    #Adam
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft,train_loss_list,val_acc_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                        num_epochs=num_epochs)
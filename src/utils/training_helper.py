import torchvision
import torch.nn as nn
import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter

def train_model(model,epochs,trainloader,validloader,criterion,optimizer,device):
    running_loss=0.0
    model.to(device)
    best_acc = 0
    for epoch in range(epochs):
        for i, data in enumerate(trainloader,0):

            model.train()
            inputs = data['ms_spot'].unsqueeze(1).permute(0,1,2,3).to(device)
            target = data['label'].to(device)
            inputs[inputs.isnan()] = 0.0
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                running_loss += loss.item()
                # print statistics


        # calculate accuracy
        with torch.no_grad():
            model.eval()
            res = torch.zeros((4,4))
            for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['ms_spot'].unsqueeze(1).permute(0,1,2,3).to(device)
                label = data['label'].to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                preds = torch.argmax(outputs,dim=1)

                for p,gt in zip(preds,label):
                    res[int(p),int(gt)] += 1




            N_total = res.sum()
            N_correct = res.diag().sum()

            acc = N_correct / N_total
            if acc > best_acc:
                print("new best acc")
                best_acc = acc
                best_model = copy.deepcopy(model) 
            print(f" Accuracy : {acc}")
        
    print('Finished Training')
    return best_model,best_acc

import torchvision
import torch.optim as optim
import torch.nn as nn
def generate_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    model.fc = nn.Linear(512,4,bias=True)
    #resnet.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #resnet.classifier = nn.Linear(in_features=densenet.classifier.in_features, out_features=3,bias=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    return model, criterion,optimizer

def train_model_ws(model,epochs,trainloader,validloader,criterion,optimizer,scheduler,device,tensorboard_name):
    running_loss=0.0
    model.to(device)
    best_acc = 0

    writer = SummaryWriter('runs/'+tensorboard_name)

    for epoch in range(epochs):
        scheduler.step()
        for i, data in enumerate(trainloader,0):

            model.train()
            inputs = data['ms_spot'].unsqueeze(1).permute(0,1,2,3).to(device)
            target = data['label'].to(device)
            inputs[inputs.isnan()] = 0.0
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                running_loss += loss.item()
                writer.add_scalar('training loss',
                running_loss / 2000,
                epoch * len(trainloader) + i)
                # print statistics


        # calculate accuracy
        with torch.no_grad():
            model.eval()
            res = torch.zeros((4,4))
            for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['ms_spot'].unsqueeze(1).permute(0,1,2,3).to(device)
                label = data['label'].to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                preds = torch.argmax(outputs,dim=1)

                for p,gt in zip(preds,label):
                    res[int(p),int(gt)] += 1




            N_total = res.sum()
            N_correct = res.diag().sum()

            acc = N_correct / N_total
            if acc > best_acc:
                print("new best acc")
                best_acc = acc
                best_model = copy.deepcopy(model) 

            writer.add_scalar('accuracy validation',
                acc,
                epoch * len(validloader) + i)
            print(f" Accuracy : {acc}")
        
    print('Finished Training')
    return best_model,best_acc

def evaluate_model(model,validloader,device):
    running_loss=0.0
    model.to(device)
    best_acc = 0
    with torch.no_grad():
        model.eval()
        res = torch.zeros((4,4))
        for i, data in enumerate(validloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['ms_spot'].unsqueeze(1).permute(0,1,2,3).to(device)
            label = data['label'].to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            preds = torch.argmax(outputs,dim=1)

            for p,gt in zip(preds,label):
                res[int(p),int(gt)] += 1




        N_total = res.sum()
        N_correct = res.diag().sum()

        acc = N_correct / N_total
        print(f"Accuracy : {acc}")
    
    return acc


    print('Finished Training')


def train_model_ws_combined(model,epochs,trainloader,validloader,criterion,optimizer,scheduler,device,tensorboard_name):
    running_loss=0.0
    model.to(device)
    best_acc = 0

    writer = SummaryWriter('runs/'+tensorboard_name)

    for epoch in range(epochs):
        scheduler.step()
        for i, data in enumerate(trainloader,0):

            model.train()
            inputs = data['combined_spot'].unsqueeze(1).to(device)
            target = data['label'].to(device)
            inputs[inputs.isnan()] = 0.0
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                running_loss += loss.item()
                writer.add_scalar('training loss',
                running_loss / 2000,
                epoch * len(trainloader) + i)
                # print statistics


        # calculate accuracy
        with torch.no_grad():
            model.eval()
            res = torch.zeros((4,4))
            for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['combined_spot'].unsqueeze(1).to(device)
                label = data['label'].to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                preds = torch.argmax(outputs,dim=1)

                for p,gt in zip(preds,label):
                    res[int(p),int(gt)] += 1




            N_total = res.sum()
            N_correct = res.diag().sum()

            acc = N_correct / N_total
            if acc > best_acc:
                print("new best acc")
                best_acc = acc
                best_model = copy.deepcopy(model) 

            writer.add_scalar('accuracy validation',
                acc,
                epoch * len(validloader) + i)
            print(f" Accuracy : {acc}")
        
    print('Finished Training')
    return best_model,best_acc


def train_model_ws_visual(model,epochs,trainloader,validloader,criterion,optimizer,scheduler,device,tensorboard_name):
    running_loss=0.0
    model.to(device)
    best_acc = 0

    writer = SummaryWriter('runs/'+tensorboard_name)

    for epoch in range(epochs):
        scheduler.step()
        for i, data in enumerate(trainloader,0):

            model.train()
            inputs = data['resnet_spot'].unsqueeze(1).to(device)
            target = data['label'].to(device)
            inputs[inputs.isnan()] = 0.0
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                running_loss += loss.item()
                writer.add_scalar('training loss',
                running_loss / 2000,
                epoch * len(trainloader) + i)
                # print statistics


        # calculate accuracy
        with torch.no_grad():
            model.eval()
            res = torch.zeros((4,4))
            for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['resnet_spot'].unsqueeze(1).to(device)
                label = data['label'].to(device)

                # forward + backward + optimize
                outputs = model(inputs)

                preds = torch.argmax(outputs,dim=1)

                for p,gt in zip(preds,label):
                    res[int(p),int(gt)] += 1




            N_total = res.sum()
            N_correct = res.diag().sum()

            acc = N_correct / N_total
            if acc > best_acc:
                print("new best acc")
                best_acc = acc
                best_model = copy.deepcopy(model) 

            writer.add_scalar('accuracy validation',
                acc,
                epoch * len(validloader) + i)
            print(f" Accuracy : {acc}")
        
    print('Finished Training')
    return best_model,best_acc


def evaluate_model_audio(model,dataloader,feature_name,device):
    running_loss=0.0
    model.to(device)
    best_acc = 0
    with torch.no_grad():
        model.eval()
        res = torch.zeros((4,4))
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['ms_spot'].unsqueeze(1).to(device)
            label = data['label'].to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            preds = torch.argmax(outputs,dim=1)

            for p,gt in zip(preds,label):
                res[int(p),int(gt)] += 1

        N_total = res.sum()
        N_correct = res.diag().sum()

        acc = N_correct / N_total
        print(f"Accuracy : {acc}")
    
    return acc


    print('Finished Training')


def evaluate_model(model,dataloader,feature_name,device):
    if not feature_name == 'combined_spot' or not feature_name == 'resnet_spot':
        return "Bad feature name!"
    running_loss=0.0
    model.to(device)
    best_acc = 0
    with torch.no_grad():
        model.eval()
        res = torch.zeros((4,4))
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[feature_name].unsqueeze(1).to(device)
            label = data['label'].to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            preds = torch.argmax(outputs,dim=1)

            for p,gt in zip(preds,label):
                res[int(p),int(gt)] += 1




        N_total = res.sum()
        N_correct = res.diag().sum()

        acc = N_correct / N_total
        print(f"Accuracy : {acc}")
    
    return acc


    print('Finished Training')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import dataset
import resnet

def test(model, optimizer, criterion, testloader):
    trainloss = 0
    trainloss_y = 0
    trainloss_n = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        label = labels[0]

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if label == 1:
            trainloss_y += loss.data[0]
        else:
            trainloss_n += loss.data[0]
        trainloss += loss.data[0]
    return trainloss, trainloss_y, trainloss_n

def train(model, optimizer, criterion, trainloader, testloader):
    losses = []
    test_losses = []
    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 20 == 19:
                losses.append(running_loss)
                with open('train_losses.csv','a') as f:
                    f.write("{}\n".format(running_loss))
                running_loss = 0.0
                
        print("epoch", epoch, "finished")
        testloss = test(model,optimizer,criterion,testloader)
        with open('test_losses.csv','a') as f:
            f.write("{},{},{}\n".format(*testloss))
        test_losses.append(testloss)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'test_losses': test_losses
        }
        torch.save(state, "saved_model")

    print('Finished Training')

def main():
    ds = dataset.TrophallaxisDataset(item_depth=3)
    trainset = ds.trainset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                                shuffle=True, num_workers=2)

    testset = ds.testset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=2)



    model = resnet.resnet34()
    model.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model,optimizer,criterion,trainloader,testloader)

if __name__ == "__main__":
    main()

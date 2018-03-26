from functools import reduce, partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix

import dataset
import resnet

MODELPATH = "saved_model"

def run_epoch(model, optimizer, criterion, loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    loss = 0.0

    y_true = []
    y_pred = []
    batchlens = []
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda(), volatile=not train), Variable(labels.cuda(), volatile= not train)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        if train:
            batchloss = criterion(outputs, labels)
            batchloss.backward()
            optimizer.step()
            loss += batchloss.data[0]

        y_true += [y for y in labels.data]
        y_pred += [0 if y[0] > y[1] else 1 for y in outputs.data]
        batchlens.append(len(y_true))

    print("batch lengths avg: {}, num batches: {}".format(sum(batchlens)/i,i))
    score = f1_score(y_true=y_true, y_pred=y_pred)
    conmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
            
    loss /= i
    score /= i
    return conmat, score, loss 


def csv(vals: list) -> str:
    return reduce(lambda x,y: str(x) + "," + str(y), vals)


def format_stats(conmat: np.ndarray, *args) -> str:
    return csv([*args, "[" + csv([int(x) for x in [*conmat[0], *conmat[1]]]) + "]"])


def run_training(model, optimizer, criterion, trainloader, testloader, start_epoch):
    run = partial(run_epoch, model=model, 
                  optimizer=optimizer, criterion=criterion)
    for epoch in range(start_epoch, 1000):
        print("train epoch", epoch)
        out = "[" + format_stats(*run(loader=trainloader, train=True))
        print("test epoch", epoch)
        conmat, _, score = run(loader=testloader, train=False)
        out += "," + format_stats(conmat, score) + "]\n"

        with open("train_stats.csv", "a") as f:
            f.write(out)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(state, MODELPATH)
        torch.cuda.empty_cache()


def restore(model, optimizer):
    if Path(MODELPATH).exists():
        state = torch.load(MODELPATH)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        return state["epoch"] + 1
    else:
        return 0


def main():
    ds = dataset.TrophallaxisDataset(item_depth=3, image_size=(64,64))
    trainset = ds.trainset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    testset = ds.testset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)
    model = resnet.resnet18()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch = restore(model,optimizer)

    model.cuda()
    print("starting training from epoch", epoch)
    run_training(model,optimizer,criterion,trainloader,testloader,epoch)


if __name__ == "__main__":
    main()

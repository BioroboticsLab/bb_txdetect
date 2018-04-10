from functools import reduce, partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from sklearn.metrics import f1_score, confusion_matrix

import dataset
import resnet
import alexnet

MODEL_PATH = "saved_model"
BEST_MODEL_PATH = "best_model"

def run_epoch(model, optimizer, criterion, loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    loss = 0.0

    y_true = []
    y_pred = []
    for i, data in enumerate(loader, 0):
        inputs = Variable(data[0].cuda(), volatile=not train)
        labels = Variable(data[1].cuda(), volatile=not train)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        if train:
            batchloss = criterion(outputs, labels)
            batchloss.backward()
            optimizer.step()
            loss += batchloss.data[0]

        y_true += [y for y in labels.data]
        y_pred += [0 if y[0] > y[1] else 1 for y in outputs.data]

    score = f1_score(y_true=y_true, y_pred=y_pred)
    conmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
            
    loss /= i
    return conmat, score, loss 


def csv(vals: list) -> str:
    return reduce(lambda x,y: str(x) + "," + str(y), vals)


def format_stats(conmat: np.ndarray, *args) -> str:
    return csv([*args, "[" + csv([int(x) for x in [*conmat[0], *conmat[1]]]) + "]"])


def run_training(model, optimizer, criterion, trainloader, testloader, start_epoch, start_score):
    run = partial(run_epoch, model=model, 
                  optimizer=optimizer, criterion=criterion)
    for epoch in range(start_epoch, 50):
        print("train epoch", epoch)
        out = "[" + format_stats(*run(loader=trainloader, train=True))
        print("test epoch", epoch)
        conmat, score, loss = run(loader=testloader, train=False)
        out += "," + format_stats(conmat, score) + "]\n"

        with open("train_stats.csv", "a") as f:
            f.write(out)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "score" : score
        }
        if score > start_score:
            torch.save(state, BEST_MODEL_PATH)
        torch.save(state, MODEL_PATH)


def restore(model, optimizer):
    if Path(MODEL_PATH).exists():
        state = torch.load(MODEL_PATH)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        return state["epoch"] + 1, state["score"] if "score" in state else 0
    else:
        return 0, 0


def main():
    img_size = 64
    item_depth = 17
    ds = dataset.TrophallaxisDataset(item_depth=item_depth, image_size=(img_size,img_size))
    trainset = ds.trainset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    testset = ds.testset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)
    model = resnet.resnet18(image_size=img_size, in_channels=item_depth)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch, score = restore(model,optimizer)

    model.cuda()
    print("img size:", img_size, "depth:", item_depth)
    print(criterion)
    print(optimizer)
    print(model)
    print("starting training from epoch", epoch)
    run_training(model,optimizer,criterion,trainloader,testloader,epoch,score)


if __name__ == "__main__":
    main()

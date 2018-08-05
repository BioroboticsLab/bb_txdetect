from functools import reduce
from pathlib import Path
from time import time
import os
import shutil
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

import dataset
import resnet
import smaller_net
from path_constants import TRAIN_LOG, DEBUG_LOG, MODEL_PATH, BEST_MODEL_PATH, TRAIN_STATS, ARCHIVE_PATH

def _run_epoch(model, optimizer, criterion, loader, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    loss = 0.0

    y_true = []
    y_pred = []
    for i, data in enumerate(loader, 0):
        inputs = Variable(data[0].cuda(), volatile=not training)
        labels = Variable(data[1].cuda(), volatile=not training)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        if training:
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



def _csv(vals: list) -> str:
    return reduce(lambda x,y: str(x) + "," + str(y), vals)


def _format_stats(conmat: np.ndarray, *args) -> str:
    return _csv([*args, "[" + _csv([int(x) for x in [*conmat[0], *conmat[1]]]) + "]"])


def _run_training(model, optimizer, criterion, trainloader, testloader, 
                 start_epoch, start_score, save_models=False, num_epochs=50, log_path=TRAIN_LOG, stats_path=TRAIN_STATS):
    tic = time()
    for epoch in tqdm(range(start_epoch, num_epochs)):
        with open(log_path, "a") as log:
            print("epoch", epoch, "train", end=" ", file=log)
            out = "[" + _format_stats(*_run_epoch(model=model, optimizer=optimizer, 
                                                  criterion=criterion, loader=trainloader, training=True))
            print("test", end=" ", file=log)
            conmat, score, _ = _run_epoch(model=model, optimizer=optimizer, criterion=criterion, 
                                          loader=testloader, training=False)
            out += "," + _format_stats(conmat, score) + "]\n"

        with open(stats_path, "a") as f:
            f.write(out)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "score" : score
        }
        if save_models:
            if score > start_score:
                torch.save(state, BEST_MODEL_PATH)
            torch.save(state, MODEL_PATH)
        toc = time()
        with open(log_path, "a") as log:
            print("time spent:", toc - tic, "sec", file=log)
        tic = toc


def _restore(model, optimizer, model_path=MODEL_PATH):
    if not Path(model_path).exists():
        return 0, 0

    state = torch.load(model_path)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"] + 1, state["score"] if "score" in state else 0



def train(seed, rca, item_depth, auto_archive=True, clahe=False, random_rotation_max=0, 
          num_epochs=50, log_path=TRAIN_LOG, stats_path=TRAIN_STATS, batch_size=64, network=None, version="2.2"):
    tic = time()
    trainset = dataset.TrophallaxisDataset(item_depth=item_depth, 
                                           random_crop_amplitude=rca, 
                                           clahe=clahe,
                                           random_rotation_max=random_rotation_max).trainset(seed=seed)

    testset = dataset.TrophallaxisDataset(item_depth=item_depth, 
                                          random_crop_amplitude=0,
                                          clahe=clahe,
                                          random_rotation_max=0).testset(seed=seed)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    if network:
        model = network(in_channels=item_depth)
    else:
        model = smaller_net.SmallerNet4(in_channels=item_depth)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001 * 64 / batch_size)

    epoch, score = _restore(model,optimizer)

    model.cuda()
    with open(log_path, "a") as log:
        print("depth:", item_depth, file=log)
        print("seed:", seed, "rca:", rca, file=log)
        print(criterion, file=log)
        print(optimizer, file=log)
        print(model, file=log)
        print("starting training from epoch", epoch, file=log)
    _run_training(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  trainloader=trainloader,
                  testloader=testloader,
                  start_epoch=epoch,
                  start_score=score,
                  num_epochs=num_epochs,
                  log_path=log_path,
                  stats_path=stats_path)
    with open(log_path, "a") as log:
        print(time() - tic, "sec spent in total", file=log)
    if auto_archive:
        archive(net=type(model).__name__, 
                size=128, 
                depth=item_depth, 
                seed=seed, 
                rca=rca, 
                version=version, 
                random_rotation_max=random_rotation_max)



def eval_untrained_model():
    img_size = 128
    item_depth = 3
    ds = dataset.TrophallaxisDataset(item_depth=item_depth)
    trainset = ds.trainset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    testset = ds.testset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)
    model = resnet.resnet18(image_size=img_size, in_channels=item_depth)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.cuda()
    return _run_epoch(model, optimizer, criterion, testloader, training=False)


def archive(net: str, size: int, depth: int, version: str, rca: int, seed: int, random_rotation_max: int):
    version = version.replace('.', '-')
    name = "{}_{}x{}_depth{}_v{}_rotation_shuffle_rca{}_seed{}_maxangle{}".format(net, size, size, depth, 
                                                                                  version, rca, seed, 
                                                                                  random_rotation_max) 
    now = datetime.datetime.now()
    datestr = now.strftime('20%y-%m-%d-%H-%M')
    name = datestr + "_" + name
    
    subfolder = "{}/{}/".format(ARCHIVE_PATH, name)
    os.mkdir(subfolder)
    for filename in ["dataset.py", "resnet.py", "train.py", "rotation.py", "smaller_net.py"]:
        shutil.copy(filename, subfolder)
    for filename in ["train_stats.csv", "trainlog.txt"]:
        os.rename(filename, "{}/{}".format(subfolder, filename)) 


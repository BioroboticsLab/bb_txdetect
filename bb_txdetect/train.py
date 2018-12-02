from functools import reduce
from pathlib import Path
from time import time
import os
import datetime
from warnings import warn
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from bb_txdetect import dataset
from bb_txdetect import smaller_net
from bb_txdetect.path_constants import (TRAIN_LOG, MODEL_PATH, TRAIN_STATS,
                                        ARCHIVE_PATH, PARAMETERS_JSON)


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
    return reduce(lambda x, y: str(x) + "," + str(y), vals)


def _format_stats(conmat: np.ndarray, *args) -> str:
    return _csv([*args, "[" + _csv([int(x) for x in [*conmat[0],
                                                     *conmat[1]]]) + "]"])


def _run_training(model, optimizer, criterion, trainloader, testloader,
                  start_epoch, start_score, save_last_model=False,
                  num_epochs=50):
    tic = time()
    for epoch in tqdm(range(start_epoch, num_epochs)):
        out = "[" + _format_stats(*_run_epoch(model=model,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              loader=trainloader,
                                              training=True))
        conmat, score, _ = _run_epoch(model=model, optimizer=optimizer,
                                      criterion=criterion,
                                      loader=testloader, training=False)
        out += "," + _format_stats(conmat, score) + "]\n"

        with open(TRAIN_STATS, "a") as f:
            f.write(out)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "score": score
        }
        toc = time()
        with open(TRAIN_LOG, "a") as log:
            print("time spent:", toc - tic, "sec", file=log)
        tic = toc
    if save_last_model:
        torch.save(state, MODEL_PATH)


def _restore(model, optimizer):
    if not Path(MODEL_PATH).exists():
        return 0, 0

    state = torch.load(MODEL_PATH)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"] + 1, state["score"] if "score" in state else 0


def cross_validate(num_runs=10, **kwargs):
    """run train num_runs times with different seeds for cross validation"""
    if "seed" in kwargs:
        warn("seed keyword is invalid, as it gets set automatically")
        del kwargs["seed"]
    for seed in range(num_runs):
        train(seed=seed, **kwargs)


def train(seed, rca, item_depth,
          drop_frames_around_trophallaxis: bool,
          auto_archive=True, clahe=False, random_rotation_max=0,
          model_parameters=None,
          num_epochs=50, batch_size=64,
          network=None, save_last_model=False,
          always_rotate_to_first_bee=False):
    """
    train a network, save stats, maybe save the model.
    Args:
        seed: determines which events go to test and which to train,
                for cross validaiton.
        rca: random crop amplitude, amount of random crops, set to 0 to disable
        item_depth: how many images should the network see for each frame.
                if set to 3 the net sees the center frame and one frame
                before and after.
                if item_depth is high, training takes longer.
        drop_frames_around_trophallaxis: if true ignore all negative frames of
                all positive events.
        auto_archive: if true the stats files get moved automatically to an
                archive folder
        clahe: apply clahe on images
        random_rotation_max: maximum angle of random rotations
        model_parameters: for models that need additional parameters
        network: class of network that should be used
    """

    tic = time()
    trainset = dataset.TrophallaxisDataset(item_depth=item_depth,
                                           random_crop_amplitude=rca,
                                           clahe=clahe,
                                           drop_frames_around_trophallaxis=drop_frames_around_trophallaxis,
                                           random_rotation_max=random_rotation_max,
                                           always_rotate_to_first_bee=always_rotate_to_first_bee).trainset(seed=seed)

    testset = dataset.TrophallaxisDataset(item_depth=item_depth,
                                          random_crop_amplitude=0,
                                          clahe=clahe,
                                          drop_frames_around_trophallaxis=drop_frames_around_trophallaxis,
                                          random_rotation_max=0,
                                          always_rotate_to_first_bee=True).testset(seed=seed)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    if network:
        if model_parameters:
            model = network(in_channels=item_depth,
                            model_parameters=model_parameters)
        else:
            model = network(in_channels=item_depth)
    else:
        model = smaller_net.SmallerNet4(in_channels=item_depth)

    criterion = nn.CrossEntropyLoss()
#   optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001 * 64 / batch_size)

    epoch, score = _restore(model, optimizer)

    model.cuda()

    params = ExperimentParameters(net=type(model).__name__)
    params.size = [128, 128]
    params.num_channels = item_depth
    params.seed = seed
    params.rca = rca
    params.version = 2.5
    params.maxangle = random_rotation_max
    params.drop = "all" if drop_frames_around_trophallaxis else 0
    params.model_parameters = model_parameters
    params.num_epochs = num_epochs
    params.clahe = clahe
    params.criterion = type(criterion).__name__
    params.optimizer = type(optimizer).__name__
    params.rotations = 1 if always_rotate_to_first_bee else 2

    params.write_file()

    _run_training(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  trainloader=trainloader,
                  testloader=testloader,
                  start_epoch=epoch,
                  start_score=score,
                  num_epochs=num_epochs,
                  save_last_model=save_last_model)
    with open(TRAIN_LOG, "a") as log:
        print(time() - tic, "sec spent in total", file=log)

    if auto_archive:
        archive(params.date)


class ExperimentParameters():
    def __init__(self, net: str):
        self.date = datetime.datetime.now().strftime('20%y-%m-%d-%H-%M')
        self.net = float(net.replace("SmallerNet", "").replace("_", "."))
        self.net_classname = net

    def write_file(self):
        with open(PARAMETERS_JSON, "w+") as f:
            json.dump(self, fp=f, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)


def archive(date: str):
    subfolder = Path(ARCHIVE_PATH) / date
    subfolder.mkdir()

    for filename in [TRAIN_STATS, TRAIN_LOG, PARAMETERS_JSON]:
        assert Path(filename).exists
        os.rename(filename, str(subfolder / Path(filename).name))

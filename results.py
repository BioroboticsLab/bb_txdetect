import json
from warnings import warn
import matplotlib.pyplot as plt
from functools import reduce
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
from path_constants import ARCHIVE_PATH, TRAIN_STATS


class Experiment(object):
    def __init__(self, path=TRAIN_STATS, experiment_id=0):
        with open(path, "r") as f:
            stats = [json.loads(line) for line in f]

        parameters_path = str(Path(path).parent / "parameters.json")
        try:
            with open(parameters_path, "r") as f:
                self.parameters = json.load(f)
        except FileNotFoundError:
            warn("File: {} not found.".format(parameters_path))

        self.experiment_id = experiment_id
        self.path = path
        self.trainscores = [line[0] for line in stats]
        self.testscores = [line[3] for line in stats]
        self.end_score_test = sum(self.testscores[-3:])/3
        self.end_score_train = sum(self.trainscores[-3:])/3
        self.best_epoch = self.testscores.index(max(self.testscores))
        self.trainloss = [line[1] for line in stats]
        self.false_predictions_train = [(line[2][1] + line[2][2]) / sum(line[2]) for line in stats]
        self.false_predictions_test = [(line[4][1] + line[4][2]) / sum(line[4]) for line in stats]

    def plot(self):
        f, axarr = plt.subplots(3, 1, figsize=(8,6), sharex=True)
        axarr[0].set_title("f1 score")
        axarr[0].plot(self.trainscores, label="train")
        axarr[0].plot(self.testscores, label="test")
        axarr[1].set_title("loss")
        axarr[1].plot(self.trainloss, label="train")
        axarr[2].set_title("incorrectly classified")
        axarr[2].plot(self.false_predictions_train, label="train")
        axarr[2].plot(self.false_predictions_test, label="test")

        for ax in axarr:
            ax.legend()
        plt.show()


def _load_experiments(folder: str = ARCHIVE_PATH):
    paths = sorted(glob("{}/*/{}".format(folder, TRAIN_STATS)))
    experiments = [Experiment(path=path, experiment_id=i) for i, path in enumerate(paths)]
    return [e for e in experiments if max(e.testscores) > 0]


def get_dataframe():
    experiments = _load_experiments()
    data = {"date":[],  "net":[],  "channels":[],  "best epoch":[],  "end score":[], 
            "end score train":[],  "version":[], "rca":[], "drop":[], "seed":[],
            "maxangle":[], "model_parameters":[] }
    for e in experiments:
        data["best epoch"].append(e.best_epoch)
        data["end score"].append(e.end_score_test)
        data["end score train"].append(e.end_score_train)
        data["date"].append(e.parameters["date"])
        data["net"].append(e.parameters["net"])
        data["channels"].append(e.parameters["num_channels"])
        data["version"].append(e.parameters["version"])
        data["rca"].append(e.parameters["rca"])
        data["drop"].append(e.parameters["drop"])
        try:
            data["seed"].append(e.parameters["seed"])
        except KeyError:
            data["seed"].append(-1)
        data["maxangle"].append(e.parameters["maxangle"])
        try:
            data["model_parameters"].append(e.parameters["model_parameters"])
        except KeyError:
            data["model_parameters"].append([])
    return pd.DataFrame(data)


def plot_experiment(experiment_id=None):
    if experiment_id:
        e = [x for x in _load_experiments() if x.experiment_id == experiment_id][0]
        print('Experiment', experiment_id, e.parameters)
    else:
        try:
            e = Experiment()
        except FileNotFoundError:
            print("no running experiment")
            return
    print('testscore  last: {} best: {}'.format(e.end_score_test, max(e.testscores)))
    print('trainscore last: {}'.format(e.end_score_train))
    e.plot()


def mean_std(df):
    return df.mean()['end score'], df.std()['end score']


class CrossValidatedResult():
    def __init__(self, version: str, crop: str, angle: str, drop: str, net: str):
        self.version = version
        self.crop = crop
        self.angle = angle
        self.drop = drop
        self.net = net

        df = get_dataframe()
        df = df[df["version"] == version]
        df = df[df["rca"] == crop]
        df = df[df["maxangle"] == angle]
        df = df[df["drop"] == drop]
        df = df[df["net"] == net]


        self.valid = list(df["seed"]) == [i for i in list(range(10))]

        if not self.valid and len(df["seed"]) >= 10:
            warn("skipped CrossValidatedResult because invalid seeds: {}".format(list(df["seed"])))


        self.mean, self.std = mean_std(df)

    def __repr__(self):
        return "f1: {:.3}, std: {:.3}, v: {}, crop: {}, rotate: {:>2}, drop: {:>3}, net: {:3}".format(
            self.mean, self.std, self.version, self.crop, self.angle, self.drop, self.net )


def get_crossvalidation_results():
    # TODO the values should be taken from df and not be hard coded
    versions = [2.2, 2.3]
    crops = [0, 8]
    angles = [0, 20]
    drops = [0, "all"]
    nets = [4.0, 4.1]
    cv_results = []
    for version in versions:
        for crop in crops:
            for angle in angles:
                for drop in drops:
                    for net in nets:
                        cvr = CrossValidatedResult(version=version,
                                                   crop=crop,
                                                   angle=angle,
                                                   drop=drop,
                                                   net=net)
                        if cvr.valid:
                            cv_results.append(cvr)
    return sorted(cv_results, key=lambda a:a.mean)

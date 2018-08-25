import json
from warnings import warn
import matplotlib.pyplot as plt
from functools import reduce
from glob import glob
import pandas as pd
import numpy as np
from path_constants import ARCHIVE_PATH, TRAIN_STATS


class Experiment(object):
    def __init__(self, path=TRAIN_STATS, experiment_id=0):
        with open(path, "r") as f:
            stats = [json.loads(line) for line in f]

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
    
    def get_param(self, prefix: str, keep_prefix=False, default="0", replace_dash=False) -> str:
        for p in self.params:
            if prefix in p and p.index(prefix) == 0:
                val = p if keep_prefix else p[len(prefix):]
                return val.replace("-", ".") if replace_dash else val
        return default
        
    @property
    def name(self):
        return reduce(lambda x,y:str(x) + " " + str(y), self.params)
    
    @property
    def params(self):
        return self.path.split("/")[1].split("_")
    
    @property
    def date(self):
        return self.get_param("20", keep_prefix=True)
    
    @property
    def net(self):
        net = self.get_param("SmallerNet", replace_dash=True)
        if not net:
            net = self.get_param("resnet")
        assert net, "network parameter not found, new name may not be matched."
        return net
    
    @property
    def img_size(self):
        return self.params[2]
    
    @property
    def num_channels(self):
        return self.get_param("depth")
    
    @property
    def version(self):
        return self.get_param("v", replace_dash=True)
    
    @property
    def rca(self):
        return self.get_param("rca")
    
    @property
    def drop(self):
        return self.get_param("drop")
    
    @property
    def seed(self):
        return self.get_param("seed", default=" ")
    
    @property
    def maxangle(self):
        return self.get_param("maxangle")
    

def _load_experiments(folder: str = ARCHIVE_PATH):
    paths = sorted(glob("{}/*/{}".format(folder, TRAIN_STATS)))
    experiments = [Experiment(path=path, experiment_id=i) for i, path in enumerate(paths)]
    return [e for e in experiments if max(e.testscores) > 0 and "ignore" not in e.name]

    
def get_dataframe():
    experiments = _load_experiments()
    data = { "date":[],  "net":[],  "channels":[],  "best epoch":[],  "end score":[], 
            "end score train":[],  "version":[], "rca":[], "drop":[], "seed":[],
            "experiment_id":[], "maxangle":[] }
    for e in experiments:
        data["date"].append(e.date)
        data["net"].append(e.net)
        data["channels"].append(e.num_channels)
        data["best epoch"].append(e.best_epoch)
        data["end score"].append(e.end_score_test)
        data["end score train"].append(e.end_score_train)
        data["version"].append(e.version)
        data["rca"].append(e.rca)
        data["drop"].append(e.drop)
        data["seed"].append(e.seed)
        data["experiment_id"].append(e.experiment_id)
        data["maxangle"].append(e.maxangle)
    return pd.DataFrame(data)


def plot_experiment(experiment_id=None):
    if experiment_id:
        e = [x for x in _load_experiments() if x.experiment_id == experiment_id][0]
        print('Experiment', experiment_id, e.name)
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


        self.valid = list(df["seed"]) == [str(i) for i in list(range(10))]

        if not self.valid and len(df["seed"]) >= 10:
            warn("skipped CrossValidatedResult because invalid seeds: {}".format(list(df["seed"])))
            

        self.mean, self.std = mean_std(df)
        
    def __repr__(self):
        return "f1: {:.3}, std: {:.3}, v: {}, crop: {}, rotate: {:>2}, drop: {:>3}, net: {:3}".format(
            self.mean, self.std, self.version, self.crop, self.angle, self.drop, self.net )


def get_crossvalidation_results():
    # TODO the values should be taken from df and not be hard coded
    versions = ["2.2", "2.3"]
    crops = ["0", "8"]
    angles = ["0", "20"]
    drops = ["0", "all"]
    nets = ["4", "4.1"]
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

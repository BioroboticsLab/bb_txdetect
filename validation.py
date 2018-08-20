import json
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
from tqdm import tqdm
import pandas as pd

from path_constants import CANDIDATES_JSON_PATH, CANDIDATES_IMAGE_FOLDER_RAW, PAD, MODEL_PATH
from dataset import ValidationSet
from smaller_net import SmallerNet4, SmallerNet4_1
from get_images import save_images
from load_data import get_all_frames


def load_candidates(frame_padding_length: int = 20, json_file_path: str = CANDIDATES_JSON_PATH, output_folder: str = CANDIDATES_IMAGE_FOLDER_RAW):
    """
    load images based on json file. the images will need further preprocessing before use.
    json file format:
        [[frame_id, bee_id0, bee_id1, x1, y1, x2, y2, cam_id]]
    """
    with open(json_file_path, "r") as f:
        candidates = json.load(f)
    for i, c in enumerate(tqdm(candidates)):
        frame_id = c[0]
        bee_ids = (c[1], c[2])
        save_images(observations=get_all_frames(frame_id_begin=frame_id, 
                                                frame_id_end=frame_id, 
                                                bee_ids=bee_ids, 
                                                frame_padding_length=frame_padding_length), 
                    index=i,
                    image_folder=output_folder)
        

def validate(model_path: str, network: torch.nn.Module, item_depth: int, validation_set: ValidationSet, optimizer=None) -> pd.DataFrame:
    model = network(in_channels=item_depth)
    model.cuda()
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    state = torch.load(model_path)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    model.eval()

    results = []
    for i, data in enumerate(tqdm(validation_set)):
        inputs = Variable(data[0].unsqueeze(0).cuda(), volatile=True)
        label = data[1]
        img_path = data[2]
        optimizer.zero_grad()
        outputs = model(inputs)
        prediction = [0 if y[0] > y[1] else 1 for y in outputs.data][0]
        confidence = float(softmax(outputs)[0][prediction])
        results.append({"label": label, "prediction": prediction, "confidence": confidence, "path": img_path})

    return pd.DataFrame(results)


def validate_pad_16() -> pd.DataFrame:
    candidates = "{}{}{}".format(CANDIDATES_IMAGE_FOLDER_RAW, PAD, 16)
    validation_set = ValidationSet(item_depth=3, img_folder=candidates)
    return validate(model_path="saved_model_sn4_1", network=SmallerNet4_1, item_depth=3, validation_set=validation_set)


def validate_depth3() -> pd.DataFrame:
    candidates = "{}{}".format(CANDIDATES_IMAGE_FOLDER_RAW, "_depth3_pad_16")
    validation_set = ValidationSet(item_depth=3, img_folder=candidates)
    return validate(model_path="saved_model_sn4_1", network=SmallerNet4_1, item_depth=3, validation_set=validation_set)

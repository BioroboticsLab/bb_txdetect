from path_constants import CANDIDATES_JSON_PATH, CANDIDATES_IMAGE_FOLDER_RAW


def load_candidates(frame_padding_length: int = 20, json_file_path: str = CANDIDATES_JSON_PATH, output_folder: str = CANDIDATES_IMAGE_FOLDER_RAW):
    """
    load images based on json file. the images will need further preprocessing before use.
    json file format:
        [[frame_id, bee_id0, bee_id1, x1, y1, x2, y2, cam_id]]
    """
    with open(json_file_path, "r") as f:
        candidates = json.load(f)
    for i in range(21,200):
        c = candidates[i]
        frame_id = c[0]
        bee_ids = (c[1], c[2])
        save_images(observations=get_all_frames(frame_id_begin=frame_id, 
                                                frame_id_end=frame_id, 
                                                bee_ids=bee_ids, 
                                                frame_padding_length=frame_padding_length), 
                    index=i,
                    image_folder=output_folder)
        

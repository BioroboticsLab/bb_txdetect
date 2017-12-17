import map_data
from tqdm import tqdm

map_data.setSnsStyle("ticks")
db = map_data.connect()
gt_events, gt_data = map_data.load_gt_data(path='csv/short.csv')
frame_fc_map = map_data.get_frame_container_info_for_frames(
    db, list(map_data.get_all_frame_ids(gt_events)))
frame_to_fc_map = map_data.get_frame_to_fc_path_dict(frame_fc_map)
map_data.map_additional_data_to_events(gt_events, frame_to_fc_map)
map_data.map_bee_ids(db, gt_events)
map_data.map_frames_before_after(db=db, events=gt_events, num_frames=8)

for event in tqdm(gt_events):
    event.df["trophallaxis_observed"] = event.trophallaxis_observed

verified_events = list(filter(lambda x: x.trophallaxis_observed == True, gt_events))

verified = gt_data[gt_data.trophallaxis_observed == 'y']
durations = verified['trophallaxis_end_frame_nr'] - verified['trophallaxis_start_frame_nr'] + 1

print("done")

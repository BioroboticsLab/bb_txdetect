from map_data import DataMapper, setSnsStyle
from tqdm import tqdm

setSnsStyle("ticks")
dm = DataMapper()
dm.connect()
dm.load_gt_data(path='csv/1.csv')
frame_fc_map = dm.get_frame_container_info_for_frames(list(dm.get_all_frame_ids()))
frame_to_fc_map = dm.get_frame_to_fc_path_dict(frame_fc_map)
dm.map_additional_data_to_events(frame_to_fc_map)
dm.map_bee_ids()
dm.map_frames_before_after(num_frames=8)

#
#for event in tqdm(gt_events):
#    event.df["trophallaxis_observed"] = event.trophallaxis_observed
#
#verified_events = list(filter(lambda x: x.trophallaxis_observed == True, gt_events))
#
#verified = gt_data[gt_data.trophallaxis_observed == 'y']
#durations = verified['trophallaxis_end_frame_nr'] - verified['trophallaxis_start_frame_nr'] + 1
#
#print("done")

from map_data import DataMapper, setSnsStyle

setSnsStyle("ticks")
dm = DataMapper()
dm.connect()
dm.load_gt_data(path='csv/ground_truth_concat.csv')
#dm.load_gt_data(path='csv/1.csv')
frame_fc_map = dm.get_frame_container_info_for_frames(list(dm.get_all_frame_ids()))
frame_to_fc_map = dm.get_frame_to_fc_path_dict(frame_fc_map)
dm.map_additional_data_to_events(frame_to_fc_map)
dm.map_bee_ids()
dm.map_observations(padding=8)

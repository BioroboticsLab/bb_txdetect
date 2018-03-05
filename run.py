from map_data import DataMapper
from get_images import save_images
dm = DataMapper(path='csv/ground_truth_concat.csv', hide_progress_bars=True)

for i in range(len(dm.gt_events)):
    save_images(dm.gt_events[i].observations, index=i)

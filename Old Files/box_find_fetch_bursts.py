import numpy as np
import os
import glob
import pandas as pd
import your
from tqdm import trange

lilo_list = glob.glob('/data/hewitt/sharing/59898/lilo*')
lilo_dict = {}

print('Connecting the start-time to specific lilo:')
for i in trange(len(lilo_list)):
    your_object = your.Your(lilo_list[i])
    start = your_object.your_header.tstart
    lilo_dict[str(start)[:15]] = lilo_list[i]

results_csv_list = glob.glob('/data/hewitt/eclat/RNarwhal/59898/ash/results*')

results_csv = pd.read_csv(results_csv_list[0])
full_array = np.zeros(len(results_csv))
results = []

for i in range(len(results_csv_list)):
    results_csv = pd.read_csv(results_csv_list[i])
    full_array += np.array(results_csv['label'] == 1, dtype='int')

for i in range(len(full_array)):
    if full_array[i] != 0:
        results.append(np.array(results_csv)[i])

candidates = []
for i in range(len(results)):
    print(results[i][1].split('_')[2])
    candidates.append([lilo_dict[results[i][1].split('_')[2][:15]], float(results[i][1].split('_')[4]), float(results[i][1].split('_')[6]),results[i][1]])
print(candidates)
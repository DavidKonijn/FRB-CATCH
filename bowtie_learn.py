import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm

burst_csv = pd.read_csv('box_burst_parameters.csv')

data = burst_csv
predicted = np.array(burst_csv["Found Burst"]).astype('int')
truth = np.array(burst_csv["Found Burst"]).astype('int')

for i in range(len(data)):
    if data['Cap'][i] == 'Cap':
        truth[i] = 0
    if data['Real'][i] == 'Real':
        truth[i] = 1

max_accuracy = 0

for a in tqdm(np.linspace(1.5,1.8,10)):
    for b in np.linspace(1.65,1.85,10):
        for d in np.linspace(0.2,0.4,10):
            for e in np.linspace(2.9,3.1,10):

                predicted_bowtie = np.zeros(len(data))

                for i in range(len(data)):
                    if data['middle'][i] > a*data['top_right'][i] and data['middle'][i] > b*data['bottom_left'][i]:
                        if data['top_right'][i] > 0*data['top_left'][i] and data['bottom_left'][i] > d*data['bottom_right'][i]:
                            if data['middle'][i] > e*data['top_band'][i]:
                                predicted_bowtie[i] = 1

                con_mtrx = confusion_matrix(truth, predicted_bowtie)

                [[true_negative,false_positive],[false_negative,true_positive]] = con_mtrx
                accuracy = (true_positive+true_negative)/(true_positive+false_negative+true_negative+false_positive)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_matrix=con_mtrx
                    best_indices=[a,b,d,e]

print("Accuracy: {}%".format(np.round(max_accuracy*100,2)))

print(best_indices)
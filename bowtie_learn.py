import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm

burst_csv = pd.read_csv('box_burst_parameters.csv')

data = burst_csv.drop(np.linspace(2774,4838, 2065))
predicted = np.array(data["Found Burst"]).astype('int')
truth = np.array(data["Found Burst"]).astype('int')

for i in range(len(data)):
    if np.array(data['Cap'])[i] == 'Cap':
        truth[i] = 0
    if np.array(data['Real'])[i] == 'Real':
        truth[i] = 1

max_sensitivity = 0
max_accuracy = 0

for k1 in tqdm(np.linspace(1.3,1.7,5)):
    for k2 in np.linspace(1.45,1.75,5):
        for k3 in np.linspace(0.2,0.3,5):
            for k4 in np.linspace(2.5,2.8,5):

                predicted_bowtie = np.zeros(len(data))

                for i in range(len(data)):
                    if np.array(data['middle'])[i] > k1*np.array(data['top_right'])[i]:
                        if np.array(data['middle'])[i] > k2*np.array(data['bottom_left'])[i]:
                            if np.array(data['bottom_left'])[i] > k3*np.array(data['bottom_right'])[i]:
                                if np.array(data['middle'])[i] > k4*np.array(data['top_band'])[i]:
                                    predicted_bowtie[i] = 1

                con_mtrx = confusion_matrix(truth, predicted_bowtie)

                [[true_negative,false_positive],[false_negative,true_positive]] = con_mtrx
                accuracy = (true_positive+true_negative)/(true_positive+false_negative+true_negative+false_positive)
                sensitivity = (true_positive)/(true_positive+false_negative)

                if sensitivity > max_sensitivity and false_positive < 100:
                    max_sensitivity = sensitivity
                    best_matrix=con_mtrx
                    best_indices=[k1, k2, k3, k4]

print("Accuracy: {}%".format(np.round(max_accuracy*100,2)))

print(best_indices)
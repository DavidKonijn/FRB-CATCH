import pandas as pd
from box_funcs import *
import time
import h5py

outdir = '/home/konijn/box_burst/dmtime/box_dm_time'
# lilo_targets = [str(59898),str(59879),str(59870),str(59873),str(59881),str(59882),str(59883),str(59884),str(59887),str(59888),str(59889)]
lilo_targets = [str(59898), str(59879)]

df = pd.DataFrame(columns=['Lilo name', 'Cand name','FETCH found', 'Model A Probability','Model B Probability','Model C Probability','Model D Probability','Model E Probability','Model F Probability','Model G Probability','Model H Probability',])

cap_real_csv = pd.read_csv('bowtie_hourglass_burst_parameters.csv')
data = cap_real_csv
predicted = np.array(cap_real_csv["FETCH found"]).astype('int')
truth = np.array(cap_real_csv["FETCH found"]).astype('int')

for i in range(len(data)):
    if data['Cap'][i] == 'Cap':
        truth[i] = 0
    if data['Real'][i] == 'Real':
        truth[i] = 1

candidate_counter = 0
double_hits = []
double_truths = []
time_between_double_hits = []

for i in range(len(lilo_targets)):
    lilo_number = lilo_targets[i]
    burst_cands, prob_array, label_array = candidate_lilo_link(lilo_number)

    unique_lilo_list = np.unique(np.array(burst_cands)[:,0])

    for j in range(len(unique_lilo_list)):
        arrival_time = []
        arrival_index = []
        for k in range(len(burst_cands)):
            if burst_cands[k][0] == unique_lilo_list[j]:
                arrival_time.append(burst_cands[k][3].split('_')[-5])
                arrival_index.append(k)
                print(burst_cands[k][0])

        sorted_time = sorted(np.array(arrival_time).astype('float'))
        sorted_index = [x for _, x in sorted(zip(np.array(arrival_time).astype('float'), arrival_index))]
        print(sorted_time)
        print(sorted_index)
        for l in range(len(sorted_time)-1):
            if sorted_time[l+1] - sorted_time[l] < 0.132:
                cand_h5_first = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/ash/"+burst_cands[sorted_index[l]][3]
                cand_h5_second = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/ash/"+burst_cands[sorted_index[l+1]][3]

                downsample_test = 0
                with h5py.File(cand_h5_first, "r") as f:
                    basename = f.attrs["basename"]
                    if basename[-2] == str(8):
                        downsample_test += 1
                with h5py.File(cand_h5_second, "r") as f:
                    basename = f.attrs["basename"]
                    if basename[-2] == str(8):
                        downsample_test += 1

                if downsample_test == 1:

                    # print(sorted_index[l+1], sorted_index[l])
                    # print(burst_cands[sorted_index[l+1]][3], burst_cands[sorted_index[l]][3])
                    # print(np.argwhere(np.array(cap_real_csv) == burst_cands[sorted_index[l+1]][3]))

                    double_hits.append([sorted_index[l+1]+candidate_counter,sorted_index[l]+candidate_counter])
                    time_between_double_hits.append(sorted_time[l+1] - sorted_time[l])
                    double_truths.append([truth[sorted_index[l+1]+candidate_counter],truth[sorted_index[l]+candidate_counter]])

    candidate_counter += len(burst_cands)

print(double_hits)
print(time_between_double_hits)
print(double_truths)
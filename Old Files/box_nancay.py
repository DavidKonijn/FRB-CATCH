import pandas as pd
from box_funcs import *
import time

outdir = '/home/konijn/box_burst/results'
lilo_targets = [str(59898),str(59879),str(59870),str(59873),str(59881),str(59882),str(59883),str(59884),str(59887),str(59888),str(59889)]
time_start = time.time()
candidate_counter = 0

for j in range(len(lilo_targets)):
    lilo_number = lilo_targets[j]
    burst_cands, prob_array, label_array = candidate_lilo_link(lilo_number)

    for i in range(len(burst_cands)):
        print("Analysis on burst ",candidate_counter," in: ", burst_cands[i][3])
        if candidate_counter > 940:
            fil_file = burst_cands[i][0]
            start_pulse = burst_cands[i][1]
            dm = burst_cands[i][2]
            burstid = candidate_counter

            lilo_name = fil_file.split('/')[-1][:-4]
            mask_file = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/"+lilo_name+"/"+lilo_name+"_badchans.txt"

            #check if the models suggest the bursts are real and only select those
            if max(prob_array[i]) > 0.5:
                select = True
            else:
                select = False
                rfi_burst(burstid, fil_file, dm, start_pulse, mask_file, select)

        # existing_df = '/home/konijn/box_burst/results/box_burst_parameters.csv'#'/data1/kenzie/M81_monitoring/bursts/Jan142022/basic_properties/Jan14_burst_properties.final.csv' #name (with full path) of the csv file containing an existing df
        # df = pd.read_csv(existing_df, index_col=0)
        # # df = pd.DataFrame(columns=['Lilo name', 'Cand name','Burst time', 'Time width','Freq width', 'S/N', 'Fluence', 'Found Burst', 'Model A Probability','Model B Probability','Model C Probability','Model D Probability','Model E Probability','Model F Probability','Model G Probability','Model H Probability',])

        # df.loc[burstid,'Lilo name'] = lilo_name
        # df.loc[burstid,'Cand name'] = burst_cands[i][3]

        # if max(prob_array[i]) > 0.5:
        #     # place a box around the injected burst
        #     best_indices, dynspec, snr, off_burst, tres, fres, fluence = box_burst(burstid, fil_file, dm, start_pulse, mask_file, select)

        #     df.loc[burstid,'Burst time']=(best_indices[1]-best_indices[0])*1.6e-2
        #     df.loc[burstid,'Time width']=best_indices[1]-best_indices[0]
        #     df.loc[burstid,'Freq width']=best_indices[3]-best_indices[2]
        #     df.loc[burstid,'S/N']=snr
        #     df.loc[burstid,'Fluence']=fluence

        # else:
        #     df.loc[burstid,'Burst time']= 'N/A'
        #     df.loc[burstid,'Time width']= 'N/A'
        #     df.loc[burstid,'Freq width']= 'N/A'
        #     df.loc[burstid,'S/N']='N/A'
        #     df.loc[burstid,'Fluence']='N/A'

        # df.loc[burstid,'Found Burst'] = select

        # df.loc[burstid,'Model A Probability'] = prob_array[i][0]
        # df.loc[burstid,'Model B Probability'] = prob_array[i][1]
        # df.loc[burstid,'Model C Probability'] = prob_array[i][2]
        # df.loc[burstid,'Model D Probability'] = prob_array[i][3]
        # df.loc[burstid,'Model E Probability'] = prob_array[i][4]
        # df.loc[burstid,'Model F Probability'] = prob_array[i][5]
        # df.loc[burstid,'Model G Probability'] = prob_array[i][6]
        # df.loc[burstid,'Model H Probability'] = prob_array[i][7]

        # df.to_csv(outdir + '/box_burst_parameters.csv')

        candidate_counter += 1

time_end = time.time()

print("Analysing "+str(len(burst_cands))+" burst candidates took: "+str(np.round((time_end-time_start)/60, 2))+" minutes.")
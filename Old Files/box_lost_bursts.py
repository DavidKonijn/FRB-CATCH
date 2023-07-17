import pandas as pd
from box_funcs import *
import time

outdir = '/home/konijn/box_burst/results/rfi/'
lilo_targets = [str(59898),str(59879),str(59870),str(59873),str(59881),str(59882),str(59883),str(59884),str(59887),str(59888),str(59889)]
time_start = time.time()
candidate_counter = 0

for j in range(len(lilo_targets)):
    lilo_number =lilo_targets[j]
    burst_cands, new_selection_candidates, prob_array, label_array = candidate_lilo_link(lilo_number)

    for i in range(len(burst_cands)):
        fil_file = burst_cands[i][0]
        start_pulse = burst_cands[i][1]
        dm = burst_cands[i][2]
        burstid = candidate_counter

        lilo_name = fil_file.split('/')[-1][:-4]
        mask_file = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/"+lilo_name+"/"+lilo_name+"_badchans.txt"

        #check if the models suggest the bursts are real and only select those
        if burst_cands[i][4] in np.array(new_selection_candidates)[:,4].astype(int):
            new_select = False
        else:
            print("Analysis on RFi sample ",candidate_counter," in: ", burst_cands[i][3])
            print(prob_array[i])
            # dynspec,dynspec_off,tres,fres,begin_bin,freqs=loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file)
            # mask_chans = np.unique(np.where(dynspec== 0)[0])
            # plot_boxxed_dynspec(dynspec, dynspec,[0,1,0,1], tres, freqs,outdir, mask_chans, False, 0, str(candidate_counter)+'_RFI_burst')

        candidate_counter += 1

time_end = time.time()

print("Analysing "+str(len(burst_cands))+" RFI candidates took: "+str(np.round((time_end-time_start)/60, 2))+" minutes.")
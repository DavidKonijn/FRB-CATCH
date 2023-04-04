import pandas as pd
from box_funcs import *
import time
import h5py
import time

outdir = '/home/konijn/box_burst/dmtime/box_dm_time'
lilo_targets = [str(59898),str(59879),str(59870),str(59873),str(59881),str(59882),str(59883),str(59884),str(59887),str(59888),str(59889)]
time_start = time.time()
candidate_counter = 0

df = pd.DataFrame(columns=['Lilo name', 'Cand name','FETCH found', 'Model A Probability','Model B Probability','Model C Probability','Model D Probability','Model E Probability','Model F Probability','Model G Probability','Model H Probability',])

for j in range(len(lilo_targets)):

    lilo_number = lilo_targets[j]
    burst_cands, prob_array, label_array = candidate_lilo_link(lilo_number)

    for i in range(len(burst_cands)):
        # if candidate_counter == 1438 or candidate_counter == 1722 or candidate_counter == 1829 or candidate_counter == 2322 or candidate_counter == 3586 or candidate_counter == 4283 or candidate_counter == 4399:
        if max(prob_array[i] > 0.5):
            print("Analysis on burst ",candidate_counter," in: ", burst_cands[i][3])

            fil_file = burst_cands[i][0]
            start_pulse = burst_cands[i][1]
            dm = burst_cands[i][2]
            burstid = candidate_counter
            select = False
            changed_width = False
            downsampled = False

            if max(prob_array[i] > 0.5):
                select = True

            lilo_name = fil_file.split('/')[-1][:-4]
            mask_file = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/"+lilo_name+"/"+lilo_name+"_badchans.txt"
            candidate_h5 = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/ash/"+burst_cands[i][3]

            with h5py.File(candidate_h5, "r") as f:
                heimdall_width = f.attrs["width"]
                basename = f.attrs["basename"]
                #check if candidate was downsampled
                if basename[-2] == str(8):
                    heimdall_width *= 8
                    downsampled = True
                    print('Downsampled!')

            #load the candidate
            dynspec, StokesI_off, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file)

            #find the optimal TOA
            best_box, dm_time = dm_time_toa(arr_not_dedispersed, frequencies, dm, bt, tsamp)

            #box the candidate
            best_indices, dynspec, snr, tres, fres, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, select, outdir, begin_t, downsampled, plot = True)

            #find the dm_trial distances
            dmtrial_rfi_distance = find_dm_trial_distance(best_indices, frequencies, dm, 0.1)
            dmtrial_height = find_dm_trial_distance(best_indices, frequencies, dm, 0.95)

            #label the candidate
            rfi_top_band, real_burst, max_power_middle = dm_time_analysis(arr_not_dedispersed, best_indices, dmtrial_rfi_distance, dmtrial_height, frequencies, dm, bt, tsamp, burstid,outdir, begin_t, plot = True)

            print('RFI:', rfi_top_band)
            print('Middle:', max_power_middle)

            df.loc[burstid,'Model A Probability'] = prob_array[i][0]
            df.loc[burstid,'Model B Probability'] = prob_array[i][1]
            df.loc[burstid,'Model C Probability'] = prob_array[i][2]
            df.loc[burstid,'Model D Probability'] = prob_array[i][3]
            df.loc[burstid,'Model E Probability'] = prob_array[i][4]
            df.loc[burstid,'Model F Probability'] = prob_array[i][5]
            df.loc[burstid,'Model G Probability'] = prob_array[i][6]
            df.loc[burstid,'Model H Probability'] = prob_array[i][7]

            df.loc[burstid,'FETCH found'] = select

            df.loc[burstid,'Lilo name'] = lilo_name
            df.loc[burstid,'Cand name'] = burst_cands[i][3]

            df.loc[burstid,'rfi_top_band'] = rfi_top_band
            df.loc[burstid,'middle'] = max_power_middle
            df.loc[burstid,'Bowtie found'] = real_burst
            df.loc[burstid,'Time width']=best_indices[1]-best_indices[0]
            df.loc[burstid,'Freq width']=best_indices[3]-best_indices[2]
            df.loc[burstid,'S/N']=snr
            df.loc[burstid,'Fluence']=fluence
            df.loc[burstid,'dm_trial']=dmtrial_rfi_distance
            df.loc[burstid,'bt']=bt
            df.loc[burstid,'downsampled']=downsampled

            # df.to_csv(outdir + '/fetch_trainburst_improved_burst_parameters.csv')

        candidate_counter += 1

time_end = time.time()

print("Analysing "+str(candidate_counter)+" burst candidates took: "+str(np.round((time_end-time_start)/60, 2))+" minutes.")
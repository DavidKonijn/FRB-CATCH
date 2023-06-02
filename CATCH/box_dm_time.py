import pandas as pd
from box_funcs import *
import h5py
import argparse

parser = argparse.ArgumentParser(description="""Creates a csv, and candidates of all potential bursts.""")
parser.add_argument('-f','--frb',type=str,required=True)
parser.add_argument('-d','--dateobs',type=int,required=True)
args = parser.parse_args()

outdir = '/data/konijn/R117_analysis/candidates'
lilo_frb = args.frb
lilo_targets = [str(args.dateobs)]
df = pd.DataFrame(columns=['lilo name', 'cand name',])

candidate_counter = 0

for j in range(len(lilo_targets)):
    lilo_number = lilo_targets[j]

    #identify each Heimdall candidate belonging to a specific file
    burst_cands, prob_array = candidate_lilo_link(lilo_number,lilo_frb)

    if burst_cands == 'Empty File':
        continue

    #remove candidates within 100ms of eachother
    burst_cands, prob_array = remove_duplicate_candidates(burst_cands, prob_array, lilo_number,lilo_frb)

    for i in range(len(burst_cands)):
        print("Analysis on burst ",candidate_counter," in: ", burst_cands[i][0][-17:], 'in', burst_cands[i][3])

        burstid = candidate_counter
        fil_file = burst_cands[i][0]
        start_pulse = float(burst_cands[i][1])
        dm =  float(burst_cands[i][2])

        fetch_prediction = False
        bowtie_prediction = False
        downsampled = False
        smaller_width = False

        lilo_name = fil_file.split('/')[-1][:-4]
        mask_file = "/data/hewitt/eclat/"+lilo_frb+"/"+lilo_number+"/"+lilo_name+"/"+lilo_name+"_badchans.txt"
        candidate_h5 = "/data/hewitt/eclat/"+lilo_frb+"/"+lilo_number+"/ash/"+burst_cands[i][3]

        #open the .h5 file
        with h5py.File(candidate_h5, "r") as f:
            heimdall_width = f.attrs["width"]
            basename = f.attrs["basename"]
            #check if candidate was downsampled
            if basename[-2] == str(8):
                heimdall_width *= 8
                downsampled = True
                print('Downsampled!')
            if heimdall_width > 1024:
                smaller_width = True

        #check if fetch identifies the burst as FRB
        if max(prob_array[i] > 0.5):
            fetch_prediction = True

        #load the candidate
        dynspec, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file)

        #find the optimal TOA
        best_box = dm_time_toa(arr_not_dedispersed, frequencies, dm, bt, tsamp, heimdall_width)

        #box the candidate
        best_indices, snr, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3], lilo_name, plot = False)

        #label the candidate
        power_top, power_middle, power_bottom, dm_trial_flag, smallest_power_factor = dm_time_analysis(arr_not_dedispersed, best_indices, frequencies, dm, burstid, outdir, begin_t, plot = False)

        if smallest_power_factor > 1.24311:
            bowtie_prediction = True

        if bowtie_prediction == True or fetch_prediction == True or snr > 200:
            frb_predictor = [bowtie_prediction, fetch_prediction, snr > 200]

            dynspec, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file, window=150)
            box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3], lilo_name, frb_predictor, plot = True)

        df.loc[burstid,'lilo name'] = lilo_name
        df.loc[burstid,'cand name'] = burst_cands[i][3]

        df.loc[burstid,'power top'] = power_top
        df.loc[burstid,'power middle'] = power_middle
        df.loc[burstid,'power bottom'] = power_bottom
        df.loc[burstid,'power factor'] = smallest_power_factor

        df.loc[burstid,'time width']=best_indices[1]-best_indices[0]
        df.loc[burstid,'freq width']=best_indices[3]-best_indices[2]
        df.loc[burstid,'center frequency']= (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000)
        df.loc[burstid,'S/N']=snr
        df.loc[burstid,'fluence']=fluence

        df.loc[burstid,'downsampled']=downsampled
        df.loc[burstid,'DM trial flag'] = dm_trial_flag
        df.loc[burstid,'smaller width'] = smaller_width

        df.loc[burstid,'model A Probability'] = prob_array[i][0]
        df.loc[burstid,'model B Probability'] = prob_array[i][1]
        df.loc[burstid,'model C Probability'] = prob_array[i][2]
        df.loc[burstid,'model D Probability'] = prob_array[i][3]
        df.loc[burstid,'model E Probability'] = prob_array[i][4]
        df.loc[burstid,'model F Probability'] = prob_array[i][5]
        df.loc[burstid,'model G Probability'] = prob_array[i][6]
        df.loc[burstid,'model H Probability'] = prob_array[i][7]

        df.loc[burstid,'FETCH prediction'] = fetch_prediction
        df.loc[burstid,'bowtie prediction'] = bowtie_prediction

        # df.to_csv(outdir + '/burst_parameters.csv')

        candidate_counter += 1

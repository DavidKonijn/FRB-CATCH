import pandas as pd
import numpy as np
from box_funcs import *
import time

outdir = '/data/hewitt/eclat/RNarwhal/60090/catch/'
burst_csv = pd.read_csv('burst_parameters.csv')
selected_truths = pd.read_csv('selected_burst_truths.csv')

lilo_targets = ['60090']

candidate_counter = 0

for j in range(len(lilo_targets)):
    lilo_number = lilo_targets[j]

    burst_cands, prob_array, label_array = candidate_lilo_link(lilo_number,lilo_frb)

    if burst_cands == 'Empty File':
        continue

    burst_cands, prob_array = remove_duplicate_candidates(burst_cands, prob_array, lilo_number,lilo_frb)

    for i in range(len(burst_cands)):
        if 'B' + str(candidate_counter) in np.array(selected_truths['Cand']):
            selected_burst_counter = np.argwhere(np.array(selected_truths['Cand']) == 'B' + str(candidate_counter))[0][0]

            if selected_truths['Real Burst Good Box'][selected_burst_counter] == True:
                print("Plotting real FRB on ",candidate_counter)

                burst_csv.loc[candidate_counter,'Real Burst']=True

                fil_file = burst_cands[i][0]
                start_pulse = float(burst_cands[i][1])
                dm =  float(burst_cands[i][2])
                burstid = candidate_counter
                fetch_prediction = False
                downsampled = False

                lilo_name = fil_file.split('/')[-1][:-4]
                mask_file = "/data/hewitt/eclat/"+lilo_frb+'/'+lilo_number+"/"+lilo_name+"/"+lilo_name+"_badchans.txt"
                candidate_h5 = "/data/hewitt/eclat/"+lilo_frb+'/'+lilo_number+"/ash/"+burst_cands[i][3]

                with h5py.File(candidate_h5, "r") as f:
                    heimdall_width = f.attrs["width"]
                    basename = f.attrs["basename"]
                    #check if candidate was downsampled
                    if basename[-2] == str(8):
                        heimdall_width *= 8
                        downsampled = True
                        print('Downsampled!')

                #check if fetch finds the burst
                if max(prob_array[i] > 0.5):
                    fetch_prediction = True

                dynspec, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file, window = 150)
                best_box = dm_time_toa(arr_not_dedispersed, frequencies, dm, bt, tsamp, heimdall_width)
                best_indices, snr, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3],lilo_name, plot = False, fancyplot = True)

                burst_csv.loc[burstid,'time width']=best_indices[1]-best_indices[0]
                burst_csv.loc[burstid,'freq width']=best_indices[3]-best_indices[2]
                burst_csv.loc[burstid,'center frequency']= (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000)
                burst_csv.loc[burstid,'S/N']=snr
                burst_csv.loc[burstid,'fluence']=fluence

            if type(selected_truths['Real Burst Wrong Box'][selected_burst_counter]) == str:
                burst_csv.loc[candidate_counter,'Real Burst']=True

                print("Fixing Box on ",candidate_counter)
                fil_file = burst_cands[i][0]
                start_pulse = float(burst_cands[i][1])
                dm =  float(burst_cands[i][2])
                burstid = candidate_counter
                fetch_prediction = False
                bowtie_prediction =False
                downsampled = False
                smaller_width = False
                coords = []

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
                    if heimdall_width > 1024:
                        smaller_width = True

                #check if fetch finds the burst
                if max(prob_array[i] > 0.5):
                    fetch_prediction = True

                #load the candidate
                dynspec, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file, window = 150)
                mask_chans = np.unique(np.where(dynspec== 0)[0])

                best_box = dm_time_toa(arr_not_dedispersed, frequencies, dm, bt, tsamp, heimdall_width)

                best_indices = [10*int(np.rint(float(selected_truths['Real Burst Wrong Box'][selected_burst_counter].split(',')[0][1:]))),
                                10*int(np.rint(float(selected_truths['Real Burst Wrong Box'][selected_burst_counter].split(',')[2]))),
                                int(np.rint(float(selected_truths['Real Burst Wrong Box'][selected_burst_counter].split(',')[1])/128*(128-len(mask_chans)))),
                                int(np.rint(float(selected_truths['Real Burst Wrong Box'][selected_burst_counter].split(',')[3][:-1])/128*(128-len(mask_chans))))]

                # reinject the masked channels
                for k in range(len(mask_chans)):
                    if mask_chans[k] <= best_indices[2]:
                        best_indices[2] += 1
                    if mask_chans[k]<=best_indices[3]:
                        best_indices[3] += 1

                #box the candidate
                best_indices, snr, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3], lilo_name, best_indices, dedicated_y_range = True, plot = False, fancyplot = True)

                burst_csv.loc[burstid,'time width']=best_indices[1]-best_indices[0]
                burst_csv.loc[burstid,'freq width']=best_indices[3]-best_indices[2]
                burst_csv.loc[burstid,'center frequency']= (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000)
                burst_csv.loc[burstid,'S/N']=snr
                burst_csv.loc[burstid,'fluence']=fluence

            if type(selected_truths['Extra Burst'][selected_burst_counter]) == str:
                burst_csv.loc[candidate_counter,'Real Burst']=True

                print("Finding extra burst on ",candidate_counter)

                fil_file = burst_cands[i][0]
                start_pulse = float(burst_cands[i][1])
                dm =  float(burst_cands[i][2])
                burstid = candidate_counter
                fetch_prediction = False
                bowtie_prediction =False
                downsampled = False
                smaller_width = False
                coords = []

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
                    if heimdall_width > 1024:
                        smaller_width = True

                #check if fetch finds the burst
                if max(prob_array[i] > 0.5):
                    fetch_prediction = True

                #load the candidate
                dynspec, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file, window = 150)
                best_box = dm_time_toa(arr_not_dedispersed, frequencies, dm, bt, tsamp, heimdall_width)
                box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3], lilo_name, plot = False, fancyplot = True)

                burstid = candidate_counter + 0.5
                mask_chans = np.unique(np.where(dynspec== 0)[0])

                best_indices = [10*int(np.rint(float(selected_truths['Extra Burst'][selected_burst_counter].split(',')[0][1:]))),
                                10*int(np.rint(float(selected_truths['Extra Burst'][selected_burst_counter].split(',')[2]))),
                                int(np.rint(float(selected_truths['Extra Burst'][selected_burst_counter].split(',')[1])/128*(128-len(mask_chans)))),
                                int(np.rint(float(selected_truths['Extra Burst'][selected_burst_counter].split(',')[3][:-1])/128*(128-len(mask_chans))))]

                # reinject the masked channels
                for k in range(len(mask_chans)):
                    if mask_chans[k] <= best_indices[2]:
                        best_indices[2] += 1
                    if mask_chans[k]<=best_indices[3]:
                        best_indices[3] += 1

                time_diff = (((best_box[1]+best_box[0])/2) - begin_t) - ((best_indices[1]+best_indices[0])/2)
                time_new =  float(burst_cands[i][3].split('_')[4]) - time_diff*(1.6e-5)
                best_box = [best_indices[0] + begin_t, best_indices[1] + begin_t]

                #box the candidate
                best_indices, snr, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3], lilo_name, best_indices, dedicated_y_range = True, plot = False, fancyplot= True)

                burst_csv.loc[burstid,'lilo name'] = lilo_name
                burst_csv.loc[burstid,'cand name'] = '_'.join(burst_cands[i][3].split('_')[:4] + [str(time_new)] + ['dm'] + burst_cands[i][3].split('_')[6:])

                burst_csv.loc[burstid,'time width']=best_indices[1]-best_indices[0]
                burst_csv.loc[burstid,'freq width']=best_indices[3]-best_indices[2]
                burst_csv.loc[burstid,'center frequency']= (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000)
                burst_csv.loc[burstid,'S/N']=snr
                burst_csv.loc[burstid,'fluence']=fluence
                burst_csv.loc[burstid,'FETCH prediction']=False

                burst_csv.loc[burstid,'Real Burst']=True

            if type(selected_truths['Both'][selected_burst_counter]) == str:
                print("Both fixing the box and finding extra burst on ",candidate_counter)

                fil_file = burst_cands[i][0]
                start_pulse = float(burst_cands[i][1])
                dm =  float(burst_cands[i][2])
                burstid = candidate_counter
                fetch_prediction = False
                bowtie_prediction =False
                downsampled = False
                smaller_width = False
                coords = []

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
                    if heimdall_width > 1024:
                        smaller_width = True

                #check if fetch finds the burst
                if max(prob_array[i] > 0.5):
                    fetch_prediction = True

                #load the candidate
                dynspec, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t= loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file, window = 150)
                mask_chans = np.unique(np.where(dynspec== 0)[0])

                best_box = dm_time_toa(arr_not_dedispersed, frequencies, dm, bt, tsamp, heimdall_width)

                best_indices = [10*int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[0][1:]))),
                                10*int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[2]))),
                                int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[1])/128*(128-len(mask_chans)))),
                                int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[3][:-1])/128*(128-len(mask_chans))))]

                # reinject the masked channels
                for k in range(len(mask_chans)):
                    if mask_chans[k] <= best_indices[2]:
                        best_indices[2] += 1
                    if mask_chans[k]<=best_indices[3]:
                        best_indices[3] += 1

                #box the candidate
                best_indices, snr, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3],lilo_name, best_indices, dedicated_y_range = True, plot = False, fancyplot = True)

                burst_csv.loc[burstid,'time width']=best_indices[1]-best_indices[0]
                burst_csv.loc[burstid,'freq width']=best_indices[3]-best_indices[2]
                burst_csv.loc[burstid,'center frequency']= (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000)
                burst_csv.loc[burstid,'S/N']=snr
                burst_csv.loc[burstid,'fluence']=fluence
                burst_csv.loc[candidate_counter,'Real Burst']=True

                burstid = candidate_counter + 0.5

                best_indices = [10*int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[4][1:]))),
                                10*int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[6]))),
                                int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[5])/128*(128-len(mask_chans)))),
                                int(np.rint(float(selected_truths['Both'][selected_burst_counter].split(',')[7][:-1])/128*(128-len(mask_chans))))]

                # reinject the masked channels
                for k in range(len(mask_chans)):
                    if mask_chans[k] <= best_indices[2]:
                        best_indices[2] += 1
                    if mask_chans[k]<=best_indices[3]:
                        best_indices[3] += 1

                time_diff = (((best_box[1]+best_box[0])/2) - begin_t) - ((best_indices[1]+best_indices[0])/2)
                time_new =  float(burst_cands[i][3].split('_')[4]) - time_diff*(1.6e-5)
                best_box = [best_indices[0] + begin_t, best_indices[1] + begin_t]

                #box the candidate
                best_indices, snr, fluence = box_burst(burstid, dynspec, best_box, heimdall_width, tsamp, freqres, frequencies, outdir, begin_t, arr_not_dedispersed, dm, downsampled, burst_cands[i][3], lilo_name, best_indices, dedicated_y_range = True, plot = False, fancyplot = True)

                burst_csv.loc[burstid,'lilo name'] = lilo_name
                burst_csv.loc[burstid,'cand name'] = '_'.join(burst_cands[i][3].split('_')[:4] + [str(time_new)] + ['dm'] + burst_cands[i][3].split('_')[6:])

                burst_csv.loc[burstid,'time width']=best_indices[1]-best_indices[0]
                burst_csv.loc[burstid,'freq width']=best_indices[3]-best_indices[2]
                burst_csv.loc[burstid,'center frequency']= (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000)
                burst_csv.loc[burstid,'S/N']=snr
                burst_csv.loc[burstid,'fluence']=fluence
                burst_csv.loc[burstid,'FETCH prediction']=False

                burst_csv.loc[burstid,'Real Burst']=True

            burst_csv.to_csv(outdir + '/updated_burst_parameters.csv')

        candidate_counter += 1

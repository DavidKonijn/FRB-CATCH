import pandas as pd
from box_funcs import *
import time

outdir = '/home/konijn/box_burst/dmtime/time_dm_plots'
lilo_targets = [str(59898),str(59879),str(59870),str(59873),str(59881),str(59882),str(59883),str(59884),str(59887),str(59888),str(59889)]
time_start = time.time()
candidate_counter = 0

# existing_df = '/home/konijn/box_burst/dmtime/time_dm_plots/box_burst_parameters.csv'#'/data1/kenzie/M81_monitoring/bursts/Jan142022/basic_properties/Jan14_burst_properties.final.csv' #name (with full path) of the csv file containing an existing df
# df = pd.read_csv(existing_df, index_col=0)

df = pd.DataFrame(columns=['Lilo name', 'Cand name','FETCH found', 'Model A Probability','Model B Probability','Model C Probability','Model D Probability','Model E Probability','Model F Probability','Model G Probability','Model H Probability',])

for j in range(len(lilo_targets)):
    lilo_number = lilo_targets[j]

    burst_cands, prob_array, label_array = candidate_lilo_link(lilo_number)

    for i in range(len(burst_cands)):

        print("Analysis on burst ",candidate_counter," in: ", burst_cands[i][3])

        fil_file = burst_cands[i][0]
        start_pulse = burst_cands[i][1]
        dm = burst_cands[i][2]
        burstid = candidate_counter
        select = False

        lilo_name = fil_file.split('/')[-1][:-4]
        mask_file = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/"+lilo_name+"/"+lilo_name+"_badchans.txt"

        if max(prob_array[i]) > 0.5:
            select = True
        
        dynspec,dynspec_off,tres,fres,begin_bin,freqs,real_burst,top_band,second_band,third_band,middle_band,fourth_band,fifth_band,bottom_band, max_intens=loaddata(fil_file, start_pulse, DM=dm, maskfile=mask_file, dmtime_anaylsis=True, burstcounter=burstid,burstname=burst_cands[i][3])

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

        df.loc[burstid,'top_band'] = top_band
        df.loc[burstid,'second_band'] = second_band
        df.loc[burstid,'third_band'] = third_band
        df.loc[burstid,'middle_band'] = middle_band
        df.loc[burstid,'fourth_band'] = fourth_band
        df.loc[burstid,'fifth band'] = fifth_band
        df.loc[burstid,'bottom_band'] = bottom_band
        df.loc[burstid,'rfi_intens'] = max_intens

        df.to_csv(outdir + '/bowtie_hourglass_burst_parameters.csv')

        candidate_counter += 1

time_end = time.time()

print("Analysing "+str(candidate_counter)+" burst candidates took: "+str(np.round((time_end-time_start)/60, 2))+" minutes.")
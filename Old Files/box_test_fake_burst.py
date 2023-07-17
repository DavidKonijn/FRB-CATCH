import pandas as pd
import os
import argparse
import pandas as pd
from box_funcs import *

outdir = '/data/konijn/eclat/pulse_injection/test_bowtie'

df = pd.DataFrame(columns=['Lilo name',])

parser = argparse.ArgumentParser(description="""Inject a FRB-like pulse into the input filterbank data and store the h5 file.""")
parser.add_argument('-t','--FRBtime',type=float,required=True)
parser.add_argument('-f','--FRBfreq',type=float,required=True)
parser.add_argument('-l','--FRBlocation',type=float,required=True)
parser.add_argument('-n','--name',type=str,required=True)
parser.add_argument('-d','--dm',type=float,required=True)
parser.add_argument('-scints','--FRBscint',type=int,required=True)
parser.add_argument('-id','--burstid',type=int,required=True)

args = parser.parse_args()
dm = args.dm
burstid = args.burstid

try:
    os.system('rm -r inject_pulse')
except:
    print("There is no inject_pulse file")

os.system('mkdir inject_pulse')
os.chdir("inject_pulse")

print("Inject pulse:")
# inject a generated FRB into real background data
start_pulse = inject_pulse(args)

# find the rfi in the injected .fil file
os.system( "rfifind -zerodm -noclip -time 0.9 -ncpus 32 -intfrac 0.28 -o rfimask_injected_pulse injected_pulse.fil")
os.system( "scp ../*.py .")

# create the identify.sh file to identify the badchannels infected by RFI
txtfile = open('identify.sh', 'w')
txtfile.write('python /data/konijn/eclat/pulse_injection/identify_badchans_py3.py -l '+str(int(dm)-30)+' -d '+str(int(dm)+30)+' -f injected_pulse.fil')
txtfile.close()

os.system( "bash identify.sh")

fil_file = "/data/konijn/eclat/pulse_injection/inject_pulse/injected_pulse.fil"
mask_file = "/data/konijn/eclat/pulse_injection/inject_pulse/injected_pulse_badchans.txt"

print("Dm/time plot analysis:")
# dynspec,dynspec_off,tres,fres,begin_bin,freqs,real_burst,rfi_top_band,middle=loaddata(fil_file, (start_pulse*1.6e-5)+(4*args.FRBtime),
#                         DM=dm, maskfile=mask_file, dmtime_anaylsis=True, plot=True, burstcounter=burstid)

#load the candidate
dynspec, StokesI_off, tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed = loaddata(fil_file, (start_pulse*1.6e-5)+(4*args.FRBtime), DM=dm, maskfile=mask_file)

#box the candidate
best_indices, dynspec, snr, off_burst, tres, fres, fluence = box_burst(burstid, dynspec, tsamp, freqres, frequencies, bt, 'False', outdir)

#find the dm_trial distance
dm_trial = find_dm_trial_distance(best_indices, frequencies, dm)

#label the candidate
rfi_top_band, middle, real_burst = dm_time_analysis(arr_not_dedispersed, dm_trial, frequencies, dm, bt, tsamp, burstid,outdir, plot = True)


df.loc[burstid,'middle'] = middle
df.loc[burstid,'rfi_top_band'] = rfi_top_band

df.to_csv(outdir + '/test_fake_bursts.csv')

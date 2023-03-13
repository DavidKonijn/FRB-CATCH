import os
import argparse
import pandas as pd

from box_funcs import *

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

outdir = '/home/konijn/box_burst/results'

try:
    os.system('rm -r inject_pulse')
except:
    print("There is no inject_pulse file")

os.system('mkdir inject_pulse')
os.chdir("inject_pulse")

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

# place a box around the injected burst
best_indices, dynspec, snr, off_burst, tres, fres= box_burst(burstid, "injected_pulse.fil", dm, start_pulse*(1.6e-5), "injected_pulse_badchans.txt")

# calculate the fluence of the burst
fluence = fluence_calculation(best_indices, dynspec, off_burst, tres, fres)

fil_file = "injected_pulse.fil"
mask_file = "injected_pulse_badchans.txt"

existing_df = '/home/konijn/box_burst/results/box_burst_parameters.csv'#'/data1/kenzie/M81_monitoring/bursts/Jan142022/basic_properties/Jan14_burst_properties.final.csv' #name (with full path) of the csv file containing an existing df
if existing_df!=None:
    df = pd.read_csv(existing_df, index_col=0)
# else:
# df = pd.DataFrame(columns=['Burst_time', 'Time_width', 'Time_width_error','Freq_width', 'Freq_width_error', 'S/N', 'Fluence'])

df.loc[burstid,'Burst_time']=(best_indices[1]-best_indices[0])*1.6e-2

df.loc[burstid,'Time_width']=best_indices[1]-best_indices[0]
df.loc[burstid,'Time_width_error']=(best_indices[1]-best_indices[0])**0.5

df.loc[burstid,'Freq_width']=best_indices[3]-best_indices[2]
df.loc[burstid,'Freq_width_error']=(best_indices[1]-best_indices[0])**0.5

df.loc[burstid,'S/N']=snr

df.loc[burstid,'Fluence']=fluence

df.to_csv(outdir + '/box_burst_parameters.csv')
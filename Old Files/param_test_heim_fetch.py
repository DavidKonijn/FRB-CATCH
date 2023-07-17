import numpy as np
import os
import argparse
import glob
import pandas as pd

from jess.fitters import median_fitter
from scipy.stats import median_abs_deviation
from will import create, inject
from your import Your
from your.formats.filwriter import make_sigproc_object

parser = argparse.ArgumentParser(description="""Inject a FRB-like pulse into the input filterbank data and store the h5 file.""")
parser.add_argument('-t','--FRBtime',type=float,required=True)
parser.add_argument('-f','--FRBfreq',type=float,required=True)
parser.add_argument('-l','--FRBlocation',type=float,required=True)
parser.add_argument('-n','--name',type=str,required=True)
parser.add_argument('-d','--dm',type=float,required=True)
parser.add_argument('-snr','--snr',type=float,required=True)
parser.add_argument('-scints','--FRBscint',type=int,required=True)

args = parser.parse_args()

try:
    os.system('rm -r inject_pulse')
except:
    print("There is no inject_pulse file")

os.system('mkdir inject_pulse')
os.chdir("inject_pulse")
lilo = glob.glob("../data_files/"+args.name+"*")[0]
print(lilo)

empty_fil =  lilo
yr_obj = Your(empty_fil)

dm = args.dm

if dm >= 500:
    # calculate the dispersive time delay
    f1 = yr_obj.your_header.center_freq - abs(yr_obj.your_header.bw)/2
    f2 = yr_obj.your_header.center_freq + abs(yr_obj.your_header.bw)/2
    dispersive_time_delay = (4.148808e6*(f1**(-2) - f2**(-2))*dm)

    # the entire generated-filterbank time will be 5 times the dispersive delay
    dispersive_range = int(((dispersive_time_delay/1000)/yr_obj.your_header.tsamp)*5)
else:
    dispersive_range = 200000

pulse_obj = create.GaussPulse(
    sigma_times=args.FRBtime,
    sigma_freqs=args.FRBfreq,
    offsets=0,
    pulse_thetas=0,
    relative_intensities=1,
    dm=args.dm,
    center_freqs=yr_obj.your_header.center_freq-args.FRBlocation,
    tau=20,
    phi=np.pi / 3,
    spectral_index_alpha=0,
    chan_freqs=yr_obj.chan_freqs,
    tsamp=yr_obj.your_header.tsamp,
    nscint=args.FRBscint,
)

pulse = pulse_obj.sample_pulse(nsamp=int(3e5)) * args.snr

# choose a random location in noise, with enough space for 3000dm pulse
i = np.random.randint(0, int(0.8*yr_obj.your_header.nspectra),1)[0]
# start between 0 and 80% of the total DR, to atleast include 1 DRange at the end
j = np.random.randint(0, int(0.8*dispersive_range),1)[0]

print("Location of noise:", i)
print("Location of start:", j)

#inject the generated pulse object into the noise filterbank
dynamic_spectra_w_pulse = inject.inject_constant_into_file(
            yr_input=yr_obj,
                pulse=(pulse*args.snr).astype('uint64'),
                    start=i+j,
                        gulp=5000000,
                        )

# initialize the new .fil file to turn into candidate .h5
sigproc_object = make_sigproc_object(
            rawdatafile  = "injected_pulse.fil",
                fch1=yr_obj.your_header.fch1,
                    foff=yr_obj.your_header.foff,
                        nchans=yr_obj.your_header.nchans,
                            source_name=yr_obj.your_header.source_name,
                                tsamp=yr_obj.your_header.tsamp,
                                    tstart=yr_obj.your_header.tstart,
                                        )

sigproc_object.write_header("injected_pulse.fil")
sigproc_object.append_spectra(dynamic_spectra_w_pulse[i:i+dispersive_range,:], "injected_pulse.fil")

# find the rfi in the injected .fil file
os.system( "rfifind -zerodm -noclip -time 0.9 -ncpus 32 -intfrac 0.28 -o rfimask_injected_pulse injected_pulse.fil")
os.system( "scp ../*.py .")

# create the identify.sh file to identify the badchannels infected by RFI
txtfile = open('identify.sh', 'w')
txtfile.write('python identify_badchans_py3.py -l '+str(int(dm)-30)+' -d '+str(int(dm)+30)+' -f injected_pulse.fil')
txtfile.close()

# zap the identified channels and make the .h5 candidate
os.system( "bash identify.sh")
os.system( "bash summonheimdall.sh")

# create the heimcsv_line.sh file to add the low changable dm
txtfile = open('heimcsv_line.sh', 'w')
txtfile.write( "python heimcsv.py -p /data/konijn/eclat/pulse_injection/inject_pulse -d "+str(int(dm))+" -t "+str(j))
txtfile.close()

os.system( "bash heimcsv_line.sh")

df=pd.read_csv("phoenix.csv")

heim_check = False

if len(df)==1:
    heim_check = True
    os.system( "your_candmaker.py -o . -n 32 -c phoenix.csv")
    h5_name = glob.glob("*.h5")[0]
    os.system( "mv "+h5_name+" ../new_h5_candidates/"+args.name+"_"+str(args.FRBtime)+"_"+str(args.FRBfreq)+"_"+str(args.dm)+"_"+h5_name)

if len(df)<2:
    lst = [[args.name,args.FRBtime,args.FRBfreq,args.dm,args.snr,args.FRBscint,args.FRBlocation,np.round(j*1.6*10**(-5),6),heim_check]]

    df = pd.DataFrame(lst)
    df.to_csv('../complete_list.csv', mode='a', header=False, index=False)

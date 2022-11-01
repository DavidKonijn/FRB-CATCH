# David Konijn
# Pulse_injection 
# 2022

import numpy as np
import os
import argparse
import glob

from jess.fitters import median_fitter
from scipy.stats import median_abs_deviation
from will import create, inject
from your import Your
from your.formats.filwriter import make_sigproc_object

parser = argparse.ArgumentParser(description="""Inject a FRB-like pulse into the input filterbank data and store the h5 file.""")
parser.add_argument('-t','--FRBtime',type=float,required=True)
parser.add_argument('-f','--FRBfreq',type=float,required=True)
parser.add_argument('-l','--lilo',type=str,required=True)
parser.add_argument('-d','--dm',type=int,required=True)
parsed.add_argument('-snr','--snr', type=float,required=True)
args = parser.parse_args()

os.chdir("../inject_pulse")
lilo = glob.glob("../data_files/"+args.lilo +"*")[0]
print(lilo)

empty_fil =  lilo
yr_obj = Your(empty_fil)

dm = args.dm

if dm >= 170:
    # calculate the dispersive time delay
    f1 = yr_obj.your_header.center_freq - abs(yr_obj.your_header.bw)/2
    f2 = yr_obj.your_header.center_freq + abs(yr_obj.your_header.bw)/2
    dispersive_time_delay = (4.148808e6*(f1**(-2) - f2**(-2))*dm)

    # the entire generated-filterbank time will be 5 times the dispersive delay
    dispersive_range = int(((dispersive_time_delay/1000)/yr_obj.your_header.tsamp)*5)
else:
    dispersive_range = 75000

pulse_obj = create.SimpleGaussPulse(
            sigma_time=args.FRBtime,
                sigma_freq=args.FRBfreq,
                    dm=args.dm,
                        center_freq=yr_obj.your_header.center_freq,
                            tau=20,
                                phi=np.pi / 3,
                                    spectral_index_alpha=0,
                                        chan_freqs=yr_obj.chan_freqs,
                                            tsamp=yr_obj.your_header.tsamp,
                                                nscint=0,
                                                    )

pulse = pulse_obj.sample_pulse(nsamp=int(3e5))

# choose a random location in noise, with enough space for 3000dm pulse
i = np.random.randint(0, int(0.8*yr_obj.your_header.nspectra),1)[0]
# start between 0 and 80% of the total DR, to atleast include 1 DRange at the end
j = np.random.randint(0, int(0.8*dispersive_range),1)[0]

print("Location of noise:", i)
print("Location of start:", j)

#inject the generated pulse object into the noise filterbank
dynamic_spectra_w_pulse = inject.inject_constant_into_file(
            yr_input=yr_obj,
                pulse=pulse*args.snr,
                    start=i+j,
                        gulp = 5000000,
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
txtfile.write('python identify_badchans_py3.py -l '+str(dm-50)+' -d '+str(dm+50)+' -f injected_pulse.fil')
txtfile.close()

# zap the identified channels and make the .h5 candidate
os.system( "bash identify.sh")
os.system( "bash summonheimdall.sh")

# create the heimcsv_line.sh file to add the low changable dm
txtfile = open('heimcsv_line.sh', 'w')
txtfile.write( "python heimcsv.py -p /data/konijn/eclat/pulse_injection/inject_pulse -d "+str(dm))
txtfile.close()

os.system( "bash heimcsv_line.sh")
os.system( "your_candmaker.py -o ../h5_candidates -n 32 -c phoenix.csv")

# copy the h5 candidate and reset the folder
# snr_name = glob.glob("*.h5")[0][-11:][:8]
# os.system( "mv injected_pulse.fil injected_pulse_"+snr_name+".fil")
# os.system( "scp injected_pulse.fil ../h5_candidates")
# os.system( "scp *.h5 ../h5_candidates")

# os.chdir( ".." )
# os.system( "rm -r inject_pulse")
# os.system( "mkdir inject_pulse")

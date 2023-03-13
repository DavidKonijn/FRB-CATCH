import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.gridspec as gridspec
import your
import pandas as pd

from presto import filterbank
from your import Your
from tqdm import trange
from will import create, inject
from matplotlib.patches import Rectangle
from your.formats.filwriter import make_sigproc_object
from scipy import signal
import time


def box_burst(burstid, pulse, dm, start_pulse, mask, new_select):
    """ Place a box around a transient signal in the input fil file
    :param pulse: filterbank file with transient signal to place a box over
    :param dm: the dm of the puls
    :param start_pulse: the startbin of the pulse
    :param mask: the mask txt file which indicates which channels to zap
    :return:
    """
    dynspec,dynspec_off,tres,fres,begin_bin,freqs,real_burst=loaddata(pulse, start_pulse, DM=dm, maskfile=mask)
    outdir = '/home/konijn/box_burst/results/'+str(burstid)+'_'

    # remove the masked channels from the data
    mask_chans = np.unique(np.where(dynspec== 0)[0])
    deleted_channels = np.delete(dynspec, mask_chans, axis=0)

    off_burst = deleted_channels[:, len(deleted_channels[0])//2 + 300:]

    converted_snr_burst = np.zeros_like(deleted_channels)

    for fr in range(deleted_channels.shape[0]):
        converted_snr_burst[fr,:]=convert_SN(deleted_channels[fr,:],off_burst[fr,:])

    # Define kernel for convolution
    kernel = np.ones([10,20])
    # Perform 2D convolution with input data and kernel
    burst = signal.convolve2d(converted_snr_burst, kernel, mode='same')/kernel.sum()
    # select only the 2.5% brightest part of the averaged burst
    simplified_burst = np.array([burst > np.percentile(burst.flatten(),95)]).astype(int)[0]
    small_burst = np.array([burst > np.percentile(burst.flatten(),99)]).astype(int)[0]

    #load_data will always put the burst at this location, so we can easily take this as an initial guess
    burst_loc_x = 3183

    burst_xwidth = max(np.sum(small_burst, axis=1))
    burst_ywidth = max(np.sum(small_burst, axis=0))
    max_value = 0

    best_indices = [0, 0, 0, 0]
    print("Determine optimal box location:")
    for i in range(20):
        ywidth = burst_ywidth+i*4
        for j in range(20):
            xwidth = burst_xwidth+j*30
            for k in range(20):
                box_y = ywidth + 5*k

                if box_y > burst.shape[0]:
                    box_y = burst.shape[0]
                    if k > 0:
                        break

                for l in range(20):
                    box_x = (burst_loc_x-xwidth//2)+(l-10)*20
                    box_intens = np.sum(simplified_burst[box_y-ywidth:box_y,
                                        box_x:box_x+xwidth])/(xwidth*ywidth)**(0.5)

                    if box_intens >= max_value:
                        max_value = box_intens
                        best_indices = [box_x,box_x+xwidth,box_y-ywidth,box_y]

    box_x_left, box_x_right, box_y_bottom, box_y_top = best_indices
    for i in range(20):
        box_x_l = box_x_left + (i-10)*10
        for j in range(20):
            box_x_r = box_x_right + (j-10)*10
            if (box_x_r-box_x_l) > 0:
                for k in range(10):
                    box_y_b = box_y_bottom + k-5
                    for l in range(10):
                        box_y_t = box_y_top + l-5
                        if box_y_t-box_y_b > 0:
                            box_intens = np.sum(simplified_burst[box_y_b:box_y_t,box_x_l:box_x_r])/((box_x_r-box_x_l)*(box_y_t-box_y_b))**(0.5)

                            if box_intens >= max_value:
                                max_value = box_intens
                                best_indices = [box_x_l,box_x_r,box_y_b,box_y_t]

    profile_burst = np.mean(converted_snr_burst[:,best_indices[0]:best_indices[1]],axis=0)
    profile_off = np.mean(off_burst[:,0:best_indices[0]-best_indices[1]],axis=0)
    profile_burst = convert_SN(profile_burst , profile_off)

    snr = np.sum(profile_burst)/np.sqrt(best_indices[1]-best_indices[0])
    profile_burst_flux=profile_burst*radiometer(tres*1000,(best_indices[3]-best_indices[2])*fres,2,35/1.4)
    fluence = np.sum(profile_burst_flux*tres*1000)

    # reinject the masked channels
    for i in range(len(mask_chans)):
        if mask_chans[i] <= best_indices[2]:
            best_indices[2] += 1
        if mask_chans[i]<=best_indices[3]:
            best_indices[3] += 1

        simplified_burst = np.insert(simplified_burst, mask_chans[i], 0, axis=0)
        burst = np.insert(burst, mask_chans[i], 0, axis=0)
        converted_snr_burst = np.insert(converted_snr_burst, mask_chans[i], 0, axis=0)
        off_burst = np.insert(off_burst, mask_chans[i], 0, axis=0)

    plot_boxxed_dynspec(converted_snr_burst, converted_snr_burst,best_indices,tres, freqs,outdir,mask_chans, new_select, snr, 'Boxed_fulldynspec')
    plot_boxxed_dynspec(simplified_burst, converted_snr_burst,best_indices,tres, freqs,outdir,mask_chans, new_select, snr, 'Boxed_simpledynspec')

    return best_indices, dynspec, snr, off_burst, tres, fres, fluence

def rfi_burst(burstid, pulse, dm, start_pulse, mask, new_select):
    """ Place a box around a transient signal in the input fil file
    :param pulse: filterbank file with transient signal to place a box over
    :param dm: the dm of the puls
    :param start_pulse: the startbin of the pulse
    :param mask: the mask txt file which indicates which channels to zap
    :return:
    """
    dynspec,dynspec_off,tres,fres,begin_bin,freqs,real_burst,middle,top_right,top_left,bottom_right,bottom_left,max_intens=loaddata(pulse, start_pulse, DM=dm, maskfile=mask)
    outdir = '/home/konijn/box_burst/results/rfi/'+str(burstid)+'_'

    try:
        dynspec=dynspec.reshape(dynspec.shape[0], dynspec.shape[-1]//5, 5).sum(axis=2)
    except ValueError:
        print("weird_shape, idk")

    # remove the masked channels from the data
    mask_chans = np.unique(np.where(dynspec== 0)[0])
    deleted_channels = np.delete(dynspec, mask_chans, axis=0)

    off_burst = deleted_channels[:, len(deleted_channels[0])//2 + 300//5:]

    converted_snr_burst = np.zeros_like(deleted_channels)

    for fr in range(deleted_channels.shape[0]):
        converted_snr_burst[fr,:]=convert_SN(deleted_channels[fr,:],off_burst[fr,:])

    # reinject the masked channels
    for i in range(len(mask_chans)):
        converted_snr_burst = np.insert(converted_snr_burst, mask_chans[i], 0, axis=0)

    profile_burst = np.mean(converted_snr_burst[:,480:800],axis=0)
    profile_off = np.mean(off_burst[:,0:480-800],axis=0)
    profile_burst = convert_SN(profile_burst , profile_off)

    snr = np.sum(profile_burst)/np.sqrt(800-480)

    plot_boxxed_dynspec(converted_snr_burst, converted_snr_burst,[480,800,0,128],tres, freqs,outdir,mask_chans, new_select, snr, 'rfi_fulldynspec')

    return

def candidate_lilo_link(lilo_number):
    lilo_list = sorted(glob.glob('/data/hewitt/sharing/'+lilo_number+'/lilo*'))
    lilo_dict = {}

    for i in range(len(lilo_list)):
        your_object = your.Your(lilo_list[i])
        start = your_object.your_header.tstart
        lilo_dict[str(start)[:11]] = lilo_list[i]

    results_csv_list = sorted(glob.glob('/data/hewitt/eclat/RNarwhal/'+lilo_number+'/ash/results*'))

    results_csv = pd.read_csv(results_csv_list[0])
    full_array = np.zeros(len(results_csv))
    prob_array = [[] for i in range(len(full_array))]
    label_array = [[] for i in range(len(full_array))]

    all_model_candidates = []

    for i in range(len(results_csv_list)):
        results_csv = pd.read_csv(results_csv_list[i])
        full_array += np.array(results_csv['label'] == 1, dtype='int')
        prob_array = np.concatenate((prob_array, np.array(results_csv['probability']).reshape(len(results_csv),1)), axis=1)
        label_array = np.concatenate((label_array, np.array(results_csv['label']).reshape(len(results_csv),1)), axis=1)

    results_csv = np.array(results_csv)
    for i in range(len(results_csv)):
        all_model_candidates.append([lilo_dict[results_csv[i][1].split('_')[2][:11]], float(results_csv[i][1].split('_')[4]), float(results_csv[i][1].split('_')[6]),results_csv[i][1], results_csv[i][0]])

    return all_model_candidates, prob_array, label_array

def plot_boxxed_dynspec(imshow, converted_snr_burst,best_indices,tres,freqs,outdir,mask_chans, new_select, snr, name):
    fig = plt.figure(figsize=(12, 12))
    rows=2
    cols=2
    widths = [3, 1]
    heights = [1,3]

    gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    time_array = (np.sum(converted_snr_burst, axis=0))/converted_snr_burst.shape[0]
    time_array_box = (np.sum(converted_snr_burst[best_indices[2]:best_indices[3],:], axis=0))/(best_indices[3]-best_indices[2])
    freq_array = np.sum(converted_snr_burst, axis=1)/converted_snr_burst.shape[1]

    ax1 = fig.add_subplot(gs[0,0]) # Time profile in S/N units
    ax1.plot(time_array, color='gray', linestyle='-', alpha=0.6, label='Full time array')
    ax1.plot(time_array_box, color='k', linestyle='-', label='Time array of the burst')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.axvline(best_indices[0],color='r', linestyle='--')
    ax1.axvline(best_indices[1],color='r', linestyle='--')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks(np.linspace(0,len(time_array),9))
    ax1.set_xticklabels((np.linspace(0,len(time_array),9)*tres*1000).astype(int).astype(str))
    ax1.legend()
    if new_select:
        ax1.set_title('FETCH found the burst', fontsize = 14)
    else:
        ax1.set_title('FETCH did NOT find the burst', fontsize = 14)

    ax2 = fig.add_subplot(gs[1,0],sharex=ax1) # Dynamic spectrum
    if new_select:
        ax2.imshow(imshow, aspect='auto', vmin = np.percentile(converted_snr_burst, 15), vmax = np.percentile(converted_snr_burst, 95))
    else:
        ax2.imshow(imshow, aspect='auto', vmin = np.percentile(converted_snr_burst, 15), vmax = np.percentile(converted_snr_burst, 99))

    for i in range(len(mask_chans)):
        ax2.axhline(mask_chans[i], linewidth=4, c='w', zorder = 1)
    ax2.add_patch(Rectangle((best_indices[0], best_indices[2]),
                            best_indices[1]-best_indices[0],
                            best_indices[3]-best_indices[2], linewidth=1.5,color = 'r', fc ='none', zorder=2))
    ax2.set_xlabel('Time [ms]', fontsize = 16)
    ax2.set_ylabel('Frequency [MHz]', fontsize = 16)
    ax2.set_yticks(np.linspace(len(freqs),0,9))
    ax2.set_yticklabels(np.linspace(freqs[0],freqs[-1],9).astype(int).astype(str))

    freq_array[np.where(freq_array == 0)] = np.nan
    ax3 = fig.add_subplot(gs[1,1]) # Spectrum
    ax3.axvline(0,color='green', linestyle='--', linewidth = 0.5, label='The zero line')
    ax3.plot(freq_array,-np.arange(len(freq_array)), color='gray', alpha=0.6, linestyle='-', drawstyle='steps', label='Full freq array')
    ax3.axhline(-best_indices[2],color='r', linestyle='--')
    ax3.axhline(-best_indices[3],color='r', linestyle='--')
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    legend = ax3.legend(loc='upper center', facecolor='white',  title='SNR: '+str(np.round(snr,1)))
    legend.get_frame().set_alpha(1)
    ax3.set_ylim(-len(freqs),1)
    plt.savefig(outdir+name+'.png',format='png',dpi=80)
    plt.show()
    plt.close()

    return snr

def inject_pulse(args):
    """ Injects an FRB pulse into real background data and returns a filterbank file with the burst
    :param args: all parameters for the burst
    :return: the startbin of the pulse
    """
    lilo = glob.glob("/data/konijn/eclat/pulse_injection/data_files/"+args.name+"*")[0]
    yr_obj = Your(lilo)

    dm = args.dm

    if dm >= 500:
        # calculate the dispersive time delay
        f1 = yr_obj.your_header.center_freq - abs(yr_obj.your_header.bw)/2
        f2 = yr_obj.your_header.center_freq + abs(yr_obj.your_header.bw)/2
        dispersive_time_delay = (4.148808e6*(f1**(-2) - f2**(-2))*dm)

        # the entire generated-filterbank time will be 5 times the dispersive delay
        dispersive_range = int(((dispersive_time_delay/1000)/yr_obj.your_header.tsamp)*5)
    else:
        dispersive_range = 250000

    pulse_obj = create.GaussPulse(
        sigma_times=args.FRBtime,
        sigma_freqs=args.FRBfreq,
        offsets=0,
        pulse_thetas=np.pi,
        relative_intensities=1,
        dm=args.dm,
        center_freqs=yr_obj.your_header.center_freq-args.FRBlocation,
        tau=20,
        phi=0,
        spectral_index_alpha=0,
        chan_freqs=yr_obj.chan_freqs,
        tsamp=yr_obj.your_header.tsamp,
        nscint=args.FRBscint,
    )

    pulse = pulse_obj.sample_pulse(nsamp=int(3e5))

    # choose a random location in noise, with enough space for 3000dm pulse
    i = np.random.randint(0, int(0.8*yr_obj.your_header.nspectra),1)[0]
    # start between 0 and 80% of the total DR, to atleast include 1 DRange at the end
    j = np.random.randint(0, int(0.8*dispersive_range),1)[0]

    #inject the generated pulse object into the noise filterbank
    dynamic_spectra_w_pulse = inject.inject_constant_into_file(
                yr_input=yr_obj,
                    pulse=(pulse*2).astype('uint64'),
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

    return j

def dedispersets(original_burst, frequencies, dms):
    original_burst = original_burst.T
    nt, nf = original_burst.shape
    assert nf == len(frequencies)
    delay_time = (4148808.0*dms*(1/(frequencies[-1])**2-1/(frequencies)**2)/1000)
    delay_bins = np.round(delay_time/(1.6e-5)).astype("int64")
    ts = np.zeros(nt, dtype=np.float32)
    for ii in range(nf):
        # if np.abs(delay_bins[ii])> original_burst.shape[0]:
        #     delay_bins[ii]+= (np.abs(delay_bins[ii])//original_burst.shape[0])*original_burst.shape[0]
        ts += np.concatenate([original_burst[-delay_bins[ii]:, ii], original_burst[:-delay_bins[ii], ii]])
    return ts

def loaddata(filename, t_burst, DM=0, maskfile=None, window=100, dmtime_anaylsis=False, burstcounter = 0, burstname= 'nan', plot = False):
    """
    Reads in the filterbank file into a numpy array, applies the mask
    and outputs the burst dynamic spectrum as a numpy array.
    Inputs:
    - filename: Name of filterbank filename with full path
    - t_burst: time of the burst in seconds into the filterbank file
    - maskfile: text file containing a list of frequency channels to zap (get it from pazi command)
    - window: window in [ms] around burst to extract for analysis (default +-10ms)
    Outputs:
    - Stokes I dynamic spectrum
    - Off burst dynamic spectrum
    - time resolution in seconds
    - frequency resolution in MHz
    """

    #NOTE dsoff size has been reduced by a factor of 5 to reduce the time it takes to load the burst
    ds,extent,tsamp,begbin, arr_not_dedispersed=load_filterbank(filename,dm=DM,fullpol=False,burst_time=t_burst)
    real_burst = False

    dsoff = ds[:, len(ds[0])//2 + 300:]
    StokesI_ds = np.zeros_like(ds)
    StokesI_off = np.zeros_like(dsoff)
    not_desispersed = np.zeros_like(arr_not_dedispersed)
    #removing bandpass

    for fr in range(ds.shape[0]):
        StokesI_ds[fr,:]=convert_SN(ds[fr,:],dsoff[fr,:])
        StokesI_off[fr,:]=convert_SN(dsoff[fr,:],dsoff[fr,:])
        not_desispersed[fr,:]=convert_SN(arr_not_dedispersed[fr,:],dsoff[fr,:])

    # frequency resolution
    freqres=(extent[3]-extent[2])/ds.shape[0]
    # frequency array
    frequencies = np.linspace(extent[2],extent[3],ds.shape[0])

    if maskfile!=None:
        try:
            maskchans=np.loadtxt(maskfile,dtype='int')
        except ValueError:
             maskchans=np.loadtxt(maskfile)
             maskchans=np.array(maskchans).astype(int)
        maskchans = [StokesI_ds.shape[0]-1-x for x in maskchans]
        StokesI_ds[maskchans,:]=0
        arr_not_dedispersed[maskchans,:]=0
        StokesI_off[maskchans,:]=0

    middle=np.nan
    top_right=np.nan
    top_left=np.nan
    bottom_right=np.nan
    bottom_left=np.nan
    max_intens=np.nan

    if dmtime_anaylsis:
        dm_list = 220 + np.linspace(-100, 100, 256)
        dm_time = np.zeros((256, arr_not_dedispersed.shape[1]), dtype=np.float32)

        for ii, dm in enumerate(dm_list):
            dm_time[ii, :] = dedispersets(arr_not_dedispersed, frequencies, dms=dm)

        dm_time -= np.median(dm_time)
        dm_time /= np.max(dm_time)

        outdir = '/home/konijn/box_burst/dmtime/time_dm_plots'
        dm_heim = float(burstname.split('_')[-3])

        locx=3183
        locy=int((dm_heim-110)*(256/220))

        box_width = 2000
        box_height = 30

        top_band = []
        second_band = []
        third_band = []
        middle_band = []
        fourth_band = []
        fifth_band = []
        bottom_band = []

        top_band.append(dm_time[locy-box_height-box_height//6:locy-box_height+box_height//6,locx-box_width:locx+box_width].flatten())
        top_band = np.average(top_band)

        second_band.append(dm_time[locy-box_height//3*2-box_height//6:locy-box_height//3*2+box_height//6,locx-box_width+box_width//4:locx+box_width-box_width//4].flatten())
        second_band = np.average(second_band)

        third_band.append(dm_time[locy-box_height//3-box_height//6:locy-box_height//3+box_height//6,locx-box_width+box_width//2:locx+box_width-box_width//2].flatten())
        third_band = np.average(third_band)

        middle_band.append(dm_time[locy-box_height//6:locy+box_height//6,locx-box_width//4:locx+box_width//4].flatten())
        middle_band = np.average(middle_band)

        fourth_band.append(dm_time[locy+box_height//3-box_height//6:locy+box_height//3+box_height//6,locx-box_width+box_width//2:locx+box_width-box_width//2].flatten())
        fourth_band = np.average(fourth_band)

        fifth_band.append(dm_time[locy+box_height//3*2-box_height//6:locy+box_height//3*2+box_height//6,locx-box_width+box_width//4:locx+box_width-box_width//4].flatten())
        fifth_band = np.average(fifth_band)

        bottom_band.append(dm_time[locy+box_height-box_height//6:locy+box_height+box_height//6,locx-box_width:locx+box_width].flatten())
        bottom_band = np.average(bottom_band)

        max_intens = 0

        for i in range(13):
            average_intensity = np.average(dm_time[0:box_height//3,i*1000:(i+1)*1000])
            if average_intensity>max_intens:
                max_intens = average_intensity

        # if middle > 1.6*top_right and middle > 1.716666666*bottom_left:
        #     if top_right > 0*top_left and bottom_left > 0.26666666*bottom_right:
        #         if middle > 3*max_intens:
        #             real_burst = True

        if plot:
            plt.figure(figsize=(12,8))
            im = plt.imshow(dm_time, aspect='auto', vmin = np.percentile(dm_time, 35), vmax = np.percentile(dm_time, 85))
            plt.xlabel('Time', fontsize = 14)
            plt.yticks(np.linspace(0,256,9), labels=np.round(np.linspace(110,330,9),0))
            plt.colorbar(im)
            plt.axvline(x=locx-box_width//4, linestyle='--', color='k')
            plt.axvline(x=locx-box_width//2, linestyle='--', color='k')
            plt.axvline(x=locx+box_width//4, linestyle='--', color='k')
            plt.axvline(x=locx+box_width//2, linestyle='--', color='k')
            plt.axvline(x=locx-box_width//4*3, linestyle='--', color='k')
            plt.axvline(x=locx+box_width//4*3, linestyle='--', color='k')
            plt.axvline(x=locx, linestyle='--', color='k')
            plt.axhline(y=locy+box_height//3+box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy+box_height//3*2+box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy+box_height+box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy+box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy-box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy-box_height//3-box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy-box_height//3*2-box_height//6, linestyle='--', color='k')
            plt.axhline(y=locy-box_height-box_height//6, linestyle='--', color='k')
            plt.plot([locx-box_width, locx-box_width, locx+box_width, locx+box_width, locx-box_width],[locy-box_height-box_height//6,locy+box_height+box_height//6,locy+box_height+box_height//6,locy-box_height-box_height//6,locy-box_height-box_height//6], color='r')
            plt.ylabel('Trial DM', fontsize = 14)
            plt.savefig(outdir+'/B%s_'%burstcounter + 'DM_time' +'_Dynamic_Spectrum.pdf',format='pdf',dpi=100)
            plt.close()

    #chop out window

    binwind=int(window/(tsamp*1000.))
    begin_t=int(t_burst/tsamp)-begbin - binwind
    end_t=int(t_burst/tsamp)-begbin +binwind

    if begin_t < 0:
        begin_t = 0
    if end_t > StokesI_ds.shape[1]:
        end_t = StokesI_ds.shape[1]-1

    StokesI_ds=StokesI_ds[:,begin_t:end_t]
    begbin=begbin+begin_t

    return np.flip(StokesI_ds,axis=0), StokesI_off, tsamp, freqres, begbin, frequencies, real_burst, top_band,second_band,third_band,middle_band,fourth_band,fifth_band,bottom_band, max_intens

def plot_ds(burstid, ds, outdir, name):
    plt.figure(figsize=(12,8))
    im = plt.imshow(ds, aspect='auto', vmin = np.percentile(ds, 15), vmax = np.percentile(ds, 95))
    plt.xlabel('Time', fontsize = 14)
    plt.colorbar(im)
    plt.ylabel('Observed Frequency', fontsize = 14)
    plt.savefig(outdir+'/B%s_'%burstid + name +'_Dynamic_Spectrum.pdf',format='pdf',dpi=100)
    plt.close()

def convert_SN(burst_prof, off_prof):
    burst_prof-=np.mean(off_prof)
    off_prof-=np.mean(off_prof)
    burst_prof/=np.std(off_prof)
    return burst_prof

def radiometer(tsamp, bw, npol, SEFD):
    """
    radiometer(tsamp, bw, npol, Tsys, G):
    tsamp is the time resolution in milliseconds
    bw is the bandwidth in MHz
    npol is the number of polarizations
    Tsys is the system temperature in K (typical value for Effelsberg = 20K)
    G is the telescope gain in K/Jy (typical value for Effelsberg = 1.54K/Jy)
    """

    return (SEFD)*(1/np.sqrt((bw*1.e6)*npol*tsamp*1e-3))

def load_filterbank(filterbank_name,dm=None,fullpol=False,burst_time=None):
    fil = filterbank.FilterbankFile(filterbank_name)
    tsamp=fil.header['tsamp']

    burst_bin = int(burst_time/tsamp)

    #consider how much delay the DM would cause
    tdel=np.abs(4.149377593360996e-3*dm*((1./np.min(fil.frequencies/1000.)**2)-(1./np.max(fil.frequencies/1000.)**2))) #seconds

    if (burst_bin-int(50e-3/tsamp)+int(200e-3/tsamp) < fil.nspec) & (burst_bin-int(50e-3/tsamp) >= 0):
        spec = fil.get_spectra(burst_bin-int(50e-3/tsamp),int(2*tdel/tsamp))
        begbin=burst_bin-int(50e-3/tsamp)
    elif burst_bin-int(50e-3/tsamp) < 0:
        spec = fil.get_spectra(0,int(2*tdel/tsamp))
        begbin=0
    else:
        dur = fil.nspec - (burst_bin-int(50e-3/tsamp))
        spec = fil.get_spectra(burst_bin-int(50e-3/tsamp),dur)
        begbin=burst_bin-int(50e-3/tsamp)

    arr_not_dedispersed=np.array(list(spec.data))

    spec.dedisperse(dm)

    arr=spec.data
    if burst_time!=None:
        #chop off the end where the DM delay creates an angled edge
        arr=arr[:,:-int(tdel/tsamp)]

    if fil.header['foff'] < 0:
        #this means the band is flipped
        arr = np.flip(arr,axis=0)
        arr_not_dedispersed = np.flip(arr_not_dedispersed,axis=0)
        foff = fil.header['foff']*-1

    else: foff = fil.header['foff']

    #header information
    tsamp = fil.header['tsamp']
    begintime = 0
    endtime = arr.shape[1]*tsamp
    fch_top = fil.header['fch1']
    nchans = fil.header['nchans']
    fch_bottom = fch_top+foff-(nchans*foff)
    extent = (begintime,endtime,fch_bottom,fch_top)

    return arr, extent, tsamp, begbin, arr_not_dedispersed
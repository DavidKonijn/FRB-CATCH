import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.gridspec as gridspec
import your
import pandas as pd
import h5py

from presto import filterbank
from your import Your
from tqdm import trange
from will import create, inject
from matplotlib.patches import Rectangle
from your.formats.filwriter import make_sigproc_object
from numba import njit

def candidate_lilo_link(lilo_number):
    """ Identify each Heimdall candidate from the inserted lilo number

    :param lilo_number: the target lilo number
    :return: Heimdall candidates and FETCH's probability array
    """

    lilo_list = sorted(glob.glob('/data/hewitt/eclat/RNarwhal/'+lilo_number+'/lilo*/*bit.fil'))
    lilo_dict = {}

    #if the data has ben cleaned i.e. there is no results folder -> skip this lilo
    if len(lilo_list) == 0:
        print("Target number-file has no bursts; empty file.")
        return 'Empty File', 'Empty File'

    #creates a dict containing the mjd of each candidate
    for i in range(len(lilo_list)):
        your_object = your.Your(lilo_list[i])
        start = your_object.your_header.tstart
        lilo_dict[str(start)[:11]] = lilo_list[i]

    results_csv_list = sorted(glob.glob('/data/hewitt/eclat/RNarwhal/'+lilo_number+'/ash/results*'))
    results_csv = pd.read_csv(results_csv_list[0])
    prob_array = [[] for i in range(len(results_csv))]

    #obtaining FETCH probabilities of each candidate from each model
    for i in range(len(results_csv_list)):
        results_csv = pd.read_csv(results_csv_list[i])
        prob_array = np.concatenate((prob_array, np.array(results_csv['probability']).reshape(len(results_csv),1)), axis=1)

    results_csv = np.array(results_csv)
    all_model_candidates = []

    #make a list containing each candidate, location of file, location of burst, dm of burst, and their heimdall candidate name
    for i in range(len(results_csv)):
        lilo_bursttime = results_csv[i][1].split('_')[2][:11]

        #due to a specific way of rounding by Your, we need to check if the middle digits are exactly 9999, and if the final digit is 6 or greater.
        #if yes, then the 5th digit after the komma needs to increase by one...
        if results_csv[i][1].split('_')[2][11:15] == '9999' and int(results_csv[i][1].split('_')[2][-1]) > 5:
            if lilo_bursttime[-1] != '9':
                lilo_bursttime = results_csv[i][1].split('_')[2][:10] + str(int(lilo_bursttime[-1])+1)
            else:
                lilo_bursttime = results_csv[i][1].split('_')[2][:10] + '0'

        all_model_candidates.append([lilo_dict[lilo_bursttime], float(results_csv[i][1].split('_')[4]), float(results_csv[i][1].split('_')[6]),results_csv[i][1], results_csv[i][0]])

    return all_model_candidates, prob_array

def remove_duplicate_candidates(burst_cands,prob_array,lilo_number):
    """ Remove duplicate candidates from downsampled and non-downsampled data

    :param burst_cands: the full burst candidate array
    :param prob_array: FETCH's probabilities for each candididate
    :param lilo_number: the lilo number
    :return: candidates and FETCH probabilities without duplicate candidates
    """

    unique_lilo_list = np.unique(np.array(burst_cands)[:,0])
    double_hits = []
    sort_all_cands = []

    #for each lilo check for candidates in downsampled and non-downsampled data within 100ms of each other
    for j in range(len(unique_lilo_list)):
        arrival_time = []
        arrival_index = []
        for k in range(len(burst_cands)):
            if burst_cands[k][0] == unique_lilo_list[j]:
                arrival_time.append(burst_cands[k][3].split('_')[-5])
                arrival_index.append(k)

        #sort the time and index in increasing time order
        sorted_time = sorted(np.array(arrival_time).astype('float'))
        sorted_index = [x for _, x in sorted(zip(np.array(arrival_time).astype('float'), arrival_index))]

        for i in range(len(sorted_index)):
            sort_all_cands.append(sorted_index[i])

        #create the list of double hits within 100ms which need to be removed
        for l in range(len(sorted_time)-1):
            if sorted_time[l+1] - sorted_time[l] < 0.1:
                cand_h5_first = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/ash/"+burst_cands[sorted_index[l]][3]
                cand_h5_second = "/data/hewitt/eclat/RNarwhal/"+lilo_number+"/ash/"+burst_cands[sorted_index[l+1]][3]

                downsample_test = 0
                with h5py.File(cand_h5_first, "r") as f:
                    basename = f.attrs["basename"]
                    if basename[-2] == str(8):
                        downsample_test += 1
                with h5py.File(cand_h5_second, "r") as f:
                    basename = f.attrs["basename"]
                    if basename[-2] == str(8):
                        downsample_test += 1

                if downsample_test == 1:
                    double_hits.append([sorted_index[l+1],sorted_index[l]])

    sorted_burst_cands = []
    sorted_prob_arr = []

    #sort_all_cands is an index list, so this selects the burst candidates based on arrival times and sorts them
    for i in range(len(sort_all_cands)):
        sorted_burst_cands.append(burst_cands[sort_all_cands[i]])
        sorted_prob_arr.append(prob_array[sort_all_cands[i]])


    #sorts the double hits based on the first burst's time of arrival
    sorted_close_cands = np.array(sorted(double_hits, key=lambda x : x[1]))

    remove_index_list = []
    for i in range(len(sorted_close_cands[:,1])):
        remove_index_list.append(np.argwhere(np.array(sort_all_cands) == sorted_close_cands[:,1][i])[0][0])

    sorted_burst_cands = np.delete(sorted_burst_cands, remove_index_list, axis = 0)
    sorted_prob_arr = np.delete(sorted_prob_arr, remove_index_list, axis = 0)

    return sorted_burst_cands, sorted_prob_arr

@njit
def convert_SN(burst_prof, off_prof):
    """ convert_SN() adapted from K.Nimmo bbpipe: https://github.com/KenzieNimmo/bbpipe
    Removing bandpass
    :param burst_prof: dynamic spectrum of the burst
    :param off_prof: dynamic spectrum off-burst
    :return: dynamic spectrum where the bandpass is removed
    """
    burst_prof-=np.mean(off_prof)
    off_prof-=np.mean(off_prof)
    burst_prof/=np.std(off_prof)
    return burst_prof

def loaddata(filename, t_burst, DM=0, maskfile=None, window=100):
    """ loaddata() adapted from K. Nimmo. bbpipe: https://github.com/KenzieNimmo/bbpipe.
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
    ds,extent,tsamp,begbin,arr_not_dedispersed,bt=load_filterbank(filename,dm=DM,fullpol=False,burst_time=t_burst)

    dsoff = ds[:, len(ds[0])//2 + 300:]
    StokesI_ds = np.zeros_like(ds)

    #removing bandpass
    for fr in range(ds.shape[0]):
        StokesI_ds[fr,:]=convert_SN(ds[fr,:],dsoff[fr,:])

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

    return np.flip(StokesI_ds,axis=0), tsamp, freqres, begbin, frequencies, bt, arr_not_dedispersed, begin_t

def dm_time_toa(arr_not_dedispersed, frequencies, DM, bt, tsamp, heimdall_width):
    """  Find the optimal TOA based on DM-time spectrum

    :param arr_not_dedispersed: complete not-dedispersed data array of the burst
    :param frequencies: frequency range of Nançay
    :param DM: Heimdall DM of the burst
    :param bt: bursttime
    :param tsamp: time sampling
    :param heimdall_width: Heimdall width of the burst in time

    :return: two coordinates indicating the left and right side of a box around the optimal DM time
    """

    locx=int(bt/tsamp)

    #heimdall sometimes makes highly inaccurate guesses, so limit the heimdall width at 50 and 1024.
    box_width = min(max(heimdall_width, 50), 1024)
    box_height = 4

    #create the DM-time spectrum DMs ranging from DM-50 to DM+50
    dm_list = DM + np.linspace(-50, 50, 30)
    dm_time = np.zeros((30, arr_not_dedispersed.shape[1]), dtype=np.float32)

    for ii, dm in enumerate(dm_list):
        dm_time[ii, :] = dedispersets(arr_not_dedispersed, frequencies, dms=dm)

    #normalize the DM-time spectrum
    dm_time -= np.median(dm_time)
    dm_time /= np.max(dm_time)

    locy=len(dm_list)//2
    max_power_middle = 1e-5
    best_box=[locx-200,locx+200]

    #shift the dm_time box horizontally to find highest power.
    left_array = np.linspace(locx-2500,locx+2500,101)
    for i in range(101):
        left_barrier = int(left_array[i])
        right_barrier = left_barrier + box_width
        if left_barrier > 0 and right_barrier < dm_time.shape[1]:
            middle = np.average(dm_time[locy-box_height//2:locy+box_height//2,left_barrier:right_barrier].flatten())
            if middle > max_power_middle:
                max_power_middle = middle
                best_box=[left_barrier,right_barrier]

    return best_box

def box_burst(burstid, dynspec, best_box, heimdall_width, tres, fres, freqs, outdir, begin_t, arr_not_dedispersed, dm, downsampled, heimdall_name, best_indices = [], dedicated_y_range = False, plot = False, fancyplot = False):
    """ Place a box around a transient signal in the input fil file
    :param burstid: the burst id of the burst
    :param dynspec: the dynamic spectrum of the burst
    :param best_box: the optimal TOA of the burst
    :param heimdall_width: Heimdalls suggested time-burst-width
    :param tres: time resolution
    :param fres: frequency resolution
    :param freqs: Nançay's frequencies
    :param outdir: outwards directory
    :param begin_t: time of burst
    :param arr_not_dedispersed: non dedispersed dynamic spectrum of the burst
    :param dm: Heimdalls suggested DM
    :param downsampled: boolean if the sample had been downsampled or not
    :param heimdall_name: candidate name based on heimdall
    :param best_indices: update indices from the clicking step
    :param dedicated_y_range: update y range from the clicking step
    :param plot: boolean if you want to plot the burst as png with the DMtime spectrum included
    :param fancyplot: boolean if you want a full dynamic spectrum pdf plot with masked frequencies

    :return: location of the best box, snr, and fluence
    """

    # remove the masked channels from the data
    mask_chans = np.unique(np.where(dynspec== 0)[0])

    deleted_channels = np.delete(dynspec, mask_chans, axis=0)
    x_loc = int(((best_box[1]+best_box[0])/2)) - begin_t

    #check the location of the burst to identify a small off burst area
    if len(deleted_channels[0])//2 + 300 >= x_loc:
        off_burst = deleted_channels[:, len(deleted_channels[0])//2 + 500:]
    else:
        off_burst = deleted_channels[:, :len(deleted_channels[0])//2 - 500]

    converted_snr_burst = np.zeros_like(deleted_channels)

    for fr in range(deleted_channels.shape[0]):
        converted_snr_burst[fr,:]=convert_SN(deleted_channels[fr,:],off_burst[fr,:])

    box_burst_dynspec = np.array(list(converted_snr_burst))

    #sometimes Heimdall has a stroke, so set the minimum at 50 and max at 1024
    heimdall_width = min(max(heimdall_width, 50), 1024)
    max_value = -1e10

    #set the width as Heimdall suggests and loop through y-axis frequencies values for the box width highest intensity
    #the penalty parameter is the sqrt of the size of the box
    if dedicated_y_range == False:
        for i in range(box_burst_dynspec.shape[0]//2):
            for j in range(box_burst_dynspec.shape[0]//2-i):
                #Heimdall makes mistakes in the downsampled version when width is 1024, so check for other appropriate widhts
                if downsampled and heimdall_width == 1024:
                    for k in range(3):
                        box_x_l = x_loc-(2**(k+8))
                        box_x_r = x_loc+(2**(k+8))
                        box_y_b = 0+i*2
                        box_y_t = box_burst_dynspec.shape[0]-j*2

                        box_intens=(np.sum(box_burst_dynspec[box_y_b:box_y_t,box_x_l:box_x_r])/((box_x_r-box_x_l)*(box_y_t-box_y_b))**(0.5))
                        if box_intens >= max_value:
                            max_value = box_intens
                            best_indices = [box_x_l,box_x_r,box_y_b,box_y_t]
                else:
                    box_x_l = x_loc-heimdall_width
                    box_x_r = x_loc+heimdall_width
                    box_y_b = 0+i*2
                    box_y_t = box_burst_dynspec.shape[0]-j*2

                    box_intens=(np.sum(box_burst_dynspec[box_y_b:box_y_t,box_x_l:box_x_r])/((box_x_r-box_x_l)*(box_y_t-box_y_b))**(0.5))
                    if box_intens >= max_value:
                        max_value = box_intens
                        best_indices = [box_x_l,box_x_r,box_y_b,box_y_t]

    #if we know the y-range due to clicking by hand, we can use that instead
    else:
        best_indices[2] -= len(np.where(mask_chans<best_indices[2])[0])
        best_indices[3] -= len(np.where(mask_chans<best_indices[3])[0])

    box_y_b = best_indices[2]
    box_y_t = best_indices[3]

    #identifyng the optimal x-values for the box
    if dedicated_y_range == False:
        for i in range(heimdall_width//4):
            for j in range(heimdall_width//4-i):
                box_x_l = x_loc-heimdall_width*2 + i*16
                box_x_r = x_loc+heimdall_width*2 - j*16

                box_intens=(np.sum(box_burst_dynspec[box_y_b:box_y_t,box_x_l:box_x_r])/((box_x_r-box_x_l)*(box_y_t-box_y_b))**(0.5))
                if box_intens >= max_value:
                    max_value = box_intens
                    best_indices = [box_x_l,box_x_r,box_y_b,box_y_t]
    else:
        box_x_l = best_indices[0]
        box_x_r = best_indices[1]
        max_value = 0

        for i in range(40):
            for j in range(40):
                box_x_l = best_indices[0] + 3*(i-20)
                box_x_r = best_indices[1] + 3*(j-20)

                box_intens=(np.sum(box_burst_dynspec[box_y_b:box_y_t,box_x_l:box_x_r])/((box_x_r-box_x_l)*(box_y_t-box_y_b))**(0.5))
                if box_intens >= max_value:
                    max_value = box_intens
                    best_indices = [box_x_l,box_x_r,box_y_b,box_y_t]

    profile_burst = np.mean(converted_snr_burst[best_indices[2]:best_indices[3],best_indices[0]:best_indices[1]], axis=0)
    profile_off = np.mean(off_burst[best_indices[2]:best_indices[3],best_indices[0]-best_indices[1]:], axis=0)
    profile_burst = convert_SN(profile_burst, profile_off)

    snr = np.sum(profile_burst)/np.sqrt(best_indices[1]-best_indices[0])

    #only in double candidate cases can an frb be inside the offburst spectrum. If this happens, snr will be negative
    #to fix this, we select the other side of the dynamic spectrum and pray that the snr is now positive :)
    if snr < 0:
        if len(deleted_channels[0])//2 + 300 >= x_loc:
            off_burst = deleted_channels[:, :len(deleted_channels[0])//2 - 500]
        else:
            off_burst = deleted_channels[:, len(deleted_channels[0])//2 + 500:]

        converted_snr_burst = np.zeros_like(deleted_channels)

        for fr in range(deleted_channels.shape[0]):
            converted_snr_burst[fr,:]=convert_SN(deleted_channels[fr,:],off_burst[fr,:])

        profile_burst = np.mean(converted_snr_burst[best_indices[2]:best_indices[3],best_indices[0]:best_indices[1]], axis=0)
        profile_off = np.mean(off_burst[best_indices[2]:best_indices[3],best_indices[0]-best_indices[1]:], axis=0)
        profile_burst = convert_SN(profile_burst, profile_off)

        snr = np.sum(profile_burst)/np.sqrt(best_indices[1]-best_indices[0])

    #use the radiometer equation to get the flux and calculate the fluence
    profile_burst_flux=profile_burst*radiometer(tres*1000,(best_indices[3]-best_indices[2])*fres,2,35/1.4)
    fluence = np.sum(profile_burst_flux*tres*1000)

    # reinject the masked channels
    for i in range(len(mask_chans)):
        if mask_chans[i] <= best_indices[2]:
            best_indices[2] += 1
        if mask_chans[i] <= best_indices[3]:
            best_indices[3] += 1
        converted_snr_burst = np.insert(converted_snr_burst, mask_chans[i], 0, axis=0)

    box_indices = list(best_indices)

    if plot:
        plot_boxxed_dynspec(converted_snr_burst,best_indices,freqs,outdir, mask_chans, begin_t, arr_not_dedispersed, dm, heimdall_name, 'Selected Boxed Bursts', burstid)
    if fancyplot:
        frb_plot(converted_snr_burst,best_indices,freqs,outdir,mask_chans,'Full Dynamic Spectrum', burstid)

    return box_indices, snr, fluence

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

def plot_boxxed_dynspec(dynspec,best_indices,freqs,outdir,mask_chans, begin_t, arr_not_dedispersed, DM, heimdall_name, name, burstid):
    """ Plots the BOXED dynamic spectrum next to the DM-time spectrum
    :param dynspec: the dynamic spectrum of the burst
    :param best_indices: update indices from the clicking step
    :param freqs: Nançay's frequencies
    :param outdir: outwards directory
    :param mask_chans: the channels with RFI-masks
    :param begin_t: time of burst
    :param arr_not_dedispersed: non dedispersed dynamic spectrum of the burst
    :param DM: Heimdalls suggested DM
    :param name: name of the plot
    :param burstid: burst id

    :return:
    """

    # calculating the full DM-time spectrum
    dynspec_indices = best_indices
    burst_duration = ((best_indices[1]-best_indices[0])*1.6e1) # in ms
    burst_bandwidth = 4*(best_indices[3]-best_indices[2]) # in MHz
    burst_central_bandwidth = (np.flip(freqs)[(best_indices[3]+best_indices[2])//2]/1000) # in GHz

    #calculate the size of the DM-time spectrum box height (at 2x burst duration), and trial-height (at 10x burst duration)
    #by using the decrease in intensity due to the smearing of the burst for wrong DMs
    dmtrial_height  = max(3, int(2*burst_duration / (8.3 * burst_bandwidth * burst_central_bandwidth**(-3))))
    dm_trial        = max(15, int(10*burst_duration / (8.3 * burst_bandwidth * burst_central_bandwidth**(-3))))

    if dmtrial_height > int(DM/5):
        dm_trial = int(DM)
        dmtrial_height = int(DM/5)

    center_burst = np.flip(freqs)[(best_indices[3]+best_indices[2])//2]
    best_indices = np.array(best_indices) + begin_t

    #calculate the entire DM-time spectrum where the y range is determined by the trial height
    dm_list = np.linspace(DM-dm_trial-dmtrial_height//2, DM+dm_trial+dmtrial_height//2, 128)
    dm_time = dedisperse_areas_around_burst(dm_list, dmtrial_height, arr_not_dedispersed, freqs, center_burst, best_indices,DM, 'Plot')

    locy = len(dm_list)//2
    locx = int(((best_indices[1]+best_indices[0])/2)) + np.round(delta_t(DM, center_burst, freqs[-1])/(1.6e-2)).astype("int64")

    box_width = best_indices[1] - best_indices[0]
    box_height = int(dmtrial_height // ((dmtrial_height+2*dm_trial)/128))

    #downsample the dynamic spectrum by a factor of 10
    tdown = 10

    if dynspec.shape[-1]%tdown == 0:
        dynspec=dynspec.reshape(dynspec.shape[0], dynspec.shape[-1]//tdown, tdown).sum(axis=2)
        dynspec_indices[0] = int(dynspec_indices[0]/tdown)
        dynspec_indices[1] = int(dynspec_indices[1]/tdown)
    else:
        for i in range(int(10-np.floor(dynspec.shape[-1]%10))):
            dynspec = np.c_[dynspec,np.zeros(dynspec.shape[0])]

        dynspec=dynspec.reshape(dynspec.shape[0], dynspec.shape[-1]//tdown, tdown).sum(axis=2)
        dynspec_indices[0] = int(dynspec_indices[0]/tdown)
        dynspec_indices[1] = int(dynspec_indices[1]/tdown)

    #remove the masked channels
    mask_chans = np.unique(np.where(dynspec== 0)[0])
    box_burst_dynspec = np.delete(dynspec, mask_chans, axis=0)

    # reject the masked channels
    dynspec_indices_with_zapped = np.array(list(dynspec_indices))
    for i in range(len(mask_chans)):
        if mask_chans[i] <= dynspec_indices_with_zapped[2]:
            dynspec_indices[2] -= 1
        if mask_chans[i] <= dynspec_indices_with_zapped[3]:
            dynspec_indices[3] -= 1

    #plot the full dynamic spectrum, with 1d side plots, and the dm time plot
    fig = plt.figure(figsize=(16, 10))
    rows=2
    cols=4
    widths = [3, 1, 0.5, 3]
    heights = [1,3]

    gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    dead_box = np.array(list(box_burst_dynspec))
    dead_box[dynspec_indices[2]:dynspec_indices[3], dynspec_indices[0]:dynspec_indices[1]] = 0

    time_array = np.sum(dead_box, axis=0)
    time_array_box = np.sum(box_burst_dynspec[dynspec_indices[2]:dynspec_indices[3],:], axis=0)
    freq_array = np.sum(dead_box, axis=1)
    freq_array_box = np.sum(box_burst_dynspec[:,dynspec_indices[0]:dynspec_indices[1]], axis=1)

    # Time profile in S/N units
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(time_array, color='gray', linestyle='-', alpha=0.6, label='Full bandwidth timeseries')
    ax1.plot(time_array_box, color='k', linestyle='-', label='Emission bandwidth timeseries')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.axvline(dynspec_indices[0],color='r', linestyle='--')
    ax1.axvline(dynspec_indices[1],color='r', linestyle='--')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks(np.linspace(0,len(time_array),9))
    ax1.set_xlim(0, len(time_array))

    #the plot line should be at the top of the band
    dynspec_indices[2] = dynspec_indices[2] - 1
    dynspec_indices[3] = dynspec_indices[3] - 1

    # Dynamic spectrum
    ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
    ax2.imshow(box_burst_dynspec, aspect='auto', vmin = np.percentile(box_burst_dynspec, 10), vmax = np.percentile(box_burst_dynspec, 100))
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax2.plot([dynspec_indices[0], dynspec_indices[0]],[dynspec_indices[2], dynspec_indices[2]+1], linewidth=1, linestyle='--', color = 'r')
    ax2.plot([dynspec_indices[0], dynspec_indices[0]+int(box_width/10)],[dynspec_indices[2], dynspec_indices[2]], linewidth=1, linestyle='-', color = 'r')
    ax2.plot([dynspec_indices[1], dynspec_indices[1]-int(box_width/10)],[dynspec_indices[3], dynspec_indices[3]], linewidth=1, linestyle='-', color = 'r')
    ax2.plot([dynspec_indices[1], dynspec_indices[1]],[dynspec_indices[3], dynspec_indices[3]-1], linewidth=1, linestyle='--', color = 'r')
    ax2.add_patch(Rectangle((0,  box_burst_dynspec.shape[0]//2),
                            100,
                            box_burst_dynspec.shape[0]//2, linewidth=1.5, alpha = 0.3, fc ='green', color = 'green', zorder =3))
    ax2.text(25, box_burst_dynspec.shape[0]-4, 'Real Burst', bbox=props)
    ax2.add_patch(Rectangle((0,  0),
                            100,
                            box_burst_dynspec.shape[0]//2, linewidth=1.5, alpha = 0.3, fc ='red', color = 'red', zorder =3))
    ax2.text(25, 4, 'No Burst', bbox=props)
    ax2.add_patch(Rectangle((box_burst_dynspec.shape[1]-100,  box_burst_dynspec.shape[0]//2),
                            100,
                            box_burst_dynspec.shape[0]//2, linewidth=1.5, alpha = 0.3, fc ='orange', color = 'orange', zorder =3))
    ax2.text(box_burst_dynspec.shape[1]-310, box_burst_dynspec.shape[0]-4, 'Wrong Box', bbox=props)
    ax2.add_patch(Rectangle((box_burst_dynspec.shape[1]-100,  0),
                            100,
                            box_burst_dynspec.shape[0]//2, linewidth=1.5, alpha = 0.5, fc ='yellow', color = 'yellow', zorder =3))
    ax2.text(box_burst_dynspec.shape[1]-370, 4, 'Another Burst', bbox=props)
    ax2.set_xlabel('Timesteps', fontsize = 16)
    ax2.set_ylabel('Frequency [MHz]', fontsize = 16)
    ax2.set_yticks(np.linspace(box_burst_dynspec.shape[0],0,9))
    ax2.set_yticklabels(np.linspace(freqs[0],freqs[-1],9).astype(int).astype(str))
    ax2.set_ylim(len(freq_array)-1, 0)

    # Freq profile in MHz units
    ax3 = fig.add_subplot(gs[1,1])
    ax3.axvline(0,color='green', linestyle='--', linewidth = 0.5)
    ax3.plot(freq_array,-np.arange(len(freq_array)), color='gray', alpha=0.6, linestyle='-', drawstyle='steps', label='Full temporal bandwidth')
    ax3.plot(freq_array_box,-np.arange(len(freq_array_box)), color='k', linestyle='-', drawstyle='steps', label='Emission temporal bandwidth')
    ax3.axhline(-dynspec_indices[2],color='r', linestyle='--')
    ax3.axhline(-dynspec_indices[3],color='r', linestyle='--')
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.set_ylim(-len(freq_array)+1,0)

    # add small distance between dynamic spectrum and dm-time
    ax4 = fig.add_subplot(gs[1,2])
    ax4.set_visible(False)

    # DM-time plot with burst indication
    ax5 = fig.add_subplot(gs[1,3])
    ax5.imshow(dm_time, aspect='auto', vmin = np.percentile(dm_time, 40), vmax = np.percentile(dm_time, 80))
    ax5.set_xlabel('Time', fontsize = 14)
    dm0_range_left = locx-10*box_width
    dm0_range_right = locx+10*box_width
    for i in range(int((dm0_range_right-dm0_range_left)/(box_width)+1)):
        ax5.plot([dm0_range_left+i*(box_width),dm0_range_left+i*(box_width)],[128,128-box_height],linestyle='--', color='r', alpha=0.6)
        ax5.plot([dm0_range_left+i*(box_width),dm0_range_left+i*(box_width)],[0,box_height],linestyle='--', color='r', alpha=0.6)
        max_right = dm0_range_left+i*(box_width)

    ax5.plot([locx-(best_indices[1]-best_indices[0])//2, locx-(best_indices[1]-best_indices[0])//2, locx+(best_indices[1]-best_indices[0])//2, locx+(best_indices[1]-best_indices[0])//2, locx-(best_indices[1]-best_indices[0])//2],[locy-box_height//2,locy+box_height//2,locy+box_height//2,locy-box_height//2,locy-box_height//2], color='r', alpha=0.6)
    ax5.plot([dm0_range_left, dm0_range_left, max_right, max_right, dm0_range_left],[128,128-box_height,128-box_height,128,128], color='r', alpha=0.6)
    ax5.plot([dm0_range_left, dm0_range_left, max_right, max_right, dm0_range_left],[0,box_height,box_height,0,0], color='r', alpha=0.6)
    ax5.set_xlim(locx-15000, locx+15000)
    ax5.set_yticks(np.linspace(0,128,9))
    ax5.set_yticklabels(np.round(np.linspace(DM-dm_trial-dmtrial_height//2, DM+dm_trial+dmtrial_height//2,9),0))
    ax5.set_ylabel('Trial DM', fontsize = 14)
    ax5.set_ylim(128,0)
    ax5.set_title("Candidate on MJD {} and at {} seconds".format(heimdall_name.split('_')[2][:5], heimdall_name.split('_')[4][:5]))

    plt.savefig(outdir+'/B%s_'%str(burstid)+name+'.png',format='png',dpi=100)
    plt.close()

    return

def frb_plot(box_burst_dynspec,dynspec_indices,freqs,outdir,mask_chans, name, burstid):
    """ Plots the fancy version of the boxed dynamic spectrum in pdf form
    :param box_burst_dynspec: the dynamic spectrum of the burst
    :param dynspec_indices: update indices from the clicking step
    :param freqs: Nançay's frequencies
    :param outdir: outwards directory
    :param mask_chans: the channels with RFI-masks
    :param name: name of the plot
    :param burstid: burst id

    :return:
    """

    #plot the full dynamic spectrum, with 1d side plots, and the dm time plot
    fig = plt.figure(figsize=(12, 10))
    rows=2
    cols=2
    widths = [3,1]
    heights = [1,3]

    gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    dead_box = np.array(list(box_burst_dynspec))
    dead_box[dynspec_indices[2]:dynspec_indices[3], dynspec_indices[0]:dynspec_indices[1]] = 0

    time_array = np.sum(dead_box, axis=0)
    time_array_box = np.sum(box_burst_dynspec[dynspec_indices[2]:dynspec_indices[3],:], axis=0)
    freq_array = np.sum(dead_box, axis=1)
    freq_array_box = np.sum(box_burst_dynspec[:,dynspec_indices[0]:dynspec_indices[1]], axis=1)

    # Time profile in S/N units
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(time_array, color='gray', linestyle='-', alpha=0.6, label='Full bandwidth timeseries')
    ax1.plot(time_array_box, color='k', linestyle='-', label='Emission bandwidth timeseries')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.axvline(dynspec_indices[0],color='r', linestyle='--')
    ax1.axvline(dynspec_indices[1],color='r', linestyle='--')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xticks(np.linspace(0,len(time_array),9))
    ax1.set_xticklabels(np.linspace(0,len(time_array)*1.6e-2,9))
    ax1.legend(facecolor='white')

    # Dynamic spectrum
    ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
    ax2.imshow(box_burst_dynspec, aspect='auto', vmin = np.percentile(box_burst_dynspec, 30), vmax = np.percentile(box_burst_dynspec, 90))

    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax2.add_patch(Rectangle((dynspec_indices[0], dynspec_indices[2]),
                            dynspec_indices[1]-dynspec_indices[0],
                            dynspec_indices[3]-dynspec_indices[2], linewidth=1, linestyle='--', color = 'r', fc ='none', zorder=2))
    for i in range(len(mask_chans)):
        ax2.axhline(mask_chans[i], linewidth=4, c='w', zorder = 1)

    ax2.set_xlabel('Time [ms]', fontsize = 16)
    ax2.set_ylabel('Frequency [MHz]', fontsize = 16)
    ax2.set_yticks(np.linspace(box_burst_dynspec.shape[0],0,9))
    ax2.set_yticklabels(np.linspace(freqs[0],freqs[-1],9).astype(int).astype(str))

    # Freq profile in MHz units
    ax3 = fig.add_subplot(gs[1,1])
    ax3.axvline(0,color='green', linestyle='--', linewidth = 0.5)
    ax3.plot(freq_array,-np.arange(len(freq_array)), color='gray', alpha=0.6, linestyle='-', drawstyle='steps', label='Full temporal bandwidth')
    ax3.plot(freq_array_box,-np.arange(len(freq_array_box)), color='k', linestyle='-', drawstyle='steps', label='Emission temporal bandwidth')
    ax3.axhline(-dynspec_indices[2],color='r', linestyle='--')
    ax3.axhline(-dynspec_indices[3],color='r', linestyle='--')
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    if dynspec_indices[2]>20:
        legend = ax3.legend(loc='upper center', facecolor='white')
    else:
        legend = ax3.legend(loc='lower center', facecolor='white')
    legend.get_frame().set_alpha(1)
    ax3.set_ylim(-len(freqs),1)
    ax3.set_ylim(-len(freq_array),1)
    plt.savefig(outdir+'/B%s_'%burstid+name+'.pdf',format='pdf',dpi=80)
    plt.close()

    return

@njit
def dedispersets(original_burst, frequencies, dms):
    """ dedisperses the data based on given DM and frequencies
    :param original_burst: DM-time spectrum of the burst
    :param frequencies: frequncy range of Nançay
    :param dms: DM range for dedispersion
    :return: dedispersed DM-time spectrum
    """
    original_burst = original_burst.T
    nt, nf = original_burst.shape
    assert nf == len(frequencies)
    delay_time = (4148808.0*dms*(1/(frequencies[-1])**2-1/(frequencies)**2)/1000)
    delay_bins = np.rint(delay_time/(1.6e-5))
    ts = np.zeros(nt, dtype=np.float32)
    for ii in range(nf):
        if np.abs(delay_bins[ii])> original_burst.shape[0]:
            delay_bins[ii] += (np.abs(delay_bins[ii])//original_burst.shape[0])*original_burst.shape[0]
        ts += np.concatenate((original_burst[-int(delay_bins[ii]):, ii], original_burst[:-int(delay_bins[ii]), ii]))
    return ts

def dm_time_analysis(arr_not_dedispersed, best_indices, frequencies, DM, burstcounter, outdir, begin_t, plot = False):
    """ DM-time spectrum Bow tie analysis to classify FRBs
    :param arr_not_dedispersed: the dynamic spectrum of the burst
    :param best_indices: update indices from the clicking step
    :param frequencies: Nançay's frequencies
    :param DM: Heimdall DM of burst
    :param outdir: the channels with RFI-masks
    :param begin_t: name of the plot
    :param burstcounter: burst id
    :param plot: boolean to plot the DM-time spectrum
    :return: the power of the boxes in the bow tie in the areas of the DM-time spectrum
    """

    burst_duration = ((best_indices[1]-best_indices[0])*1.6e1) # in ms
    burst_bandwidth = 4*(best_indices[3]-best_indices[2]) # in MHz
    burst_central_bandwidth = (np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]/1000) # in GHz

    #calculate the size of the DM-time spectrum box height (at 2x burst duration), and trial-height (at 10x burst duration)
    #by using the decrease in intensity due to the smearing of the burst for wrong DMs
    dmtrial_height  = max(3, int(2*burst_duration / (8.3 * burst_bandwidth * burst_central_bandwidth**(-3))))
    dm_trial        = max(15, int(10*burst_duration / (8.3 * burst_bandwidth * burst_central_bandwidth**(-3))))
    dm_trial_flag   = False

    if dmtrial_height > int(DM/5):
        dm_trial = int(DM)
        dmtrial_height = int(DM/5)
        print('Dm Trials too large, reducing...')
        dm_trial_flag = True

    center_burst = np.flip(frequencies)[(best_indices[3]+best_indices[2])//2]
    best_indices = np.array(best_indices) + begin_t

    #only dedisperse in the areas used for ratio-calculation

    #top_band_power
    dm_list = np.linspace(DM-dm_trial-dmtrial_height//2, DM-dm_trial+dmtrial_height//2, dmtrial_height)
    power_top = dedisperse_areas_around_burst(dm_list, dmtrial_height, arr_not_dedispersed, frequencies, center_burst, best_indices,DM, 'Top')

    #middle power:
    dm_list = np.linspace(DM-dmtrial_height//2, DM+dmtrial_height//2, dmtrial_height)
    power_middle = dedisperse_areas_around_burst(dm_list, dmtrial_height, arr_not_dedispersed, frequencies, center_burst, best_indices,DM, 'Middle')

    #bottom_band_power
    dm_list = np.linspace(DM+dm_trial-dmtrial_height//2, DM+dm_trial+dmtrial_height//2, dmtrial_height)
    power_bottom = dedisperse_areas_around_burst(dm_list, dmtrial_height, arr_not_dedispersed, frequencies, center_burst, best_indices,DM, 'Top')

    print('Power Top:       ', power_top)
    print('Power Middle:    ', power_middle)
    print('Power Bottom:    ', power_bottom)

    print('Smallest Power Factor: ', min(power_middle/power_bottom, power_middle/power_top))

    if plot:
        #the entire spectrum needs to be dedispersed for the plot
        dm_list = np.linspace(DM-dm_trial-dmtrial_height//2, DM+dm_trial+dmtrial_height//2, 128)
        dm_time = dedisperse_areas_around_burst(dm_list, dmtrial_height, arr_not_dedispersed, frequencies, center_burst, best_indices,DM, 'Plot')

        locy = len(dm_list)//2
        locx = int(((best_indices[1]+best_indices[0])/2)) + np.round(delta_t(DM, center_burst, frequencies[-1])/(1.6e-2)).astype("int64")

        box_width = best_indices[1] - best_indices[0]
        box_height = int(dmtrial_height // ((dmtrial_height+2*dm_trial)/128))

        plt.figure(figsize=(12,8))
        im = plt.imshow(dm_time, aspect='auto', vmin = np.percentile(dm_time, 35), vmax = np.percentile(dm_time, 85))
        plt.xlabel('Time', fontsize = 14)

        dm0_range_left = locx-10*box_width
        dm0_range_right = locx+10*box_width
        for i in range(int((dm0_range_right-dm0_range_left)/(box_width)+1)):
            plt.plot([dm0_range_left+i*(box_width),dm0_range_left+i*(box_width)],[128,128-box_height],linestyle='--', color='r', alpha=0.6)
            plt.plot([dm0_range_left+i*(box_width),dm0_range_left+i*(box_width)],[0,box_height],linestyle='--', color='r', alpha=0.6)
            max_right = dm0_range_left+i*(box_width)

        plt.plot([locx-(best_indices[1]-best_indices[0])//2, locx-(best_indices[1]-best_indices[0])//2, locx+(best_indices[1]-best_indices[0])//2, locx+(best_indices[1]-best_indices[0])//2, locx-(best_indices[1]-best_indices[0])//2],[locy-box_height//2,locy+box_height//2,locy+box_height//2,locy-box_height//2,locy-box_height//2], color='r', alpha=0.6)
        plt.plot([dm0_range_left, dm0_range_left, max_right, max_right, dm0_range_left],[128,128-box_height,128-box_height,128,128], color='r', alpha=0.6)
        plt.plot([dm0_range_left, dm0_range_left, max_right, max_right, dm0_range_left],[0,box_height,box_height,0,0], color='r', alpha=0.6)
        plt.xlim(0, min(2*locx, 60000))
        plt.yticks(np.linspace(0,128,9), labels=np.round(np.linspace(DM-dm_trial-dmtrial_height//2, DM+dm_trial+dmtrial_height//2,9),0))
        plt.colorbar(im)
        plt.ylabel('Trial DM', fontsize = 14)
        plt.ylim(128,0)
        plt.savefig(outdir+'/B%s_'%burstcounter + 'DM_time.pdf',format='pdf',dpi=100)
        plt.close()

    return power_top, power_middle, power_bottom, dm_trial_flag, min(power_middle/power_bottom, power_middle/power_top)

def delta_t(DM, fref, fchan):
    #Pulsar handbook dispersive delay between two frequencies in MHz
    return 4.148808*10**6 * (fref**(-2) - fchan**(-2)) * DM

def dedisperse_areas_around_burst(dm_list, dmtrial_height, arr_not_dedispersed, frequencies, center_burst, best_indices ,DM, mid_or_top):
    """ Create the DM-time spectrum from the dispersed dynamic spectrum using the given dm_list
    :param dm_list: range of DM values
    :param dmtrial_height: height of the box placed
    :param arr_not_dedispersed: dispersed dynamic spectrum
    :param frequencies: frequency ranges of Nançay
    :param center_burst: the center burst frequency used to straighten the burst
    :best_indices: the coordinates of the box placed around the burst in dynamic spectrum, indicates the width of the burst
    :param DM: Heimdall DM of burst
    :param mid_or_top: boolean indicates if the box is placed in the middle of the dynamic spectrum or on the edges
    :return: differs between mid_or_top plot, either the maximum intensity of the burst or of the RFI
    """

    box_width = best_indices[1] - best_indices[0]
    dm_time = np.zeros((len(dm_list), arr_not_dedispersed.shape[1]), dtype=np.float32)

    for ii, dm in enumerate(dm_list):
        dm_time[ii, :] = dedispersets(arr_not_dedispersed, frequencies, dms=dm)

    dm_time -= np.median(dm_time)
    dm_time /= np.max(dm_time)

    #allign the burst vertically
    roll_list = np.rint(delta_t(dm_list, center_burst, frequencies[-1])/(1.6e-2))

    for i in range(len(dm_time)):
       dm_time[i] = np.roll(dm_time[i], int(roll_list[i]))

    locx=int(((best_indices[1]+best_indices[0])/2)) + int(np.rint(delta_t(DM, center_burst, frequencies[-1])/(1.6e-2)))
    locy = dmtrial_height//2

    if mid_or_top == 'Middle':
        max_middle = np.average(dm_time[locy-dmtrial_height//2:locy+dmtrial_height//2,locx-box_width//2:locx+box_width//2].flatten())
        returned_object = max_middle
    elif mid_or_top == 'Top' or mid_or_top == 'Bottom':
        dm0_range_left = locx-10*box_width
        dm0_range_right = locx+10*box_width
        max_rfi = 1e-4

        for i in range(int((dm0_range_right-dm0_range_left)/(box_width))):
            if int(dm0_range_left+i*(box_width)) > 0 and int(dm0_range_left+(i+1)*(box_width)) < dm_time.shape[1]:
                rfi_top_band1 = np.average(dm_time[locy-dmtrial_height//2:locy+dmtrial_height//2,int(dm0_range_left+i*(box_width)):int(dm0_range_left+(i+1)*(box_width))])
            else:
                rfi_top_band1 = 1e-6
            if int(dm0_range_left-(box_width//2)+i*(box_width)) > 0 and int(dm0_range_left-(box_width//2)+(i+1)*(box_width)) < dm_time.shape[1]:
                rfi_top_band2 = np.average(dm_time[locy-dmtrial_height//2:locy+dmtrial_height//2,int(dm0_range_left-(box_width//2)+i*(box_width)):int(dm0_range_left-(box_width//2)+(i+1)*(box_width))])
            else:
                rfi_top_band2 = 1e-6
            rfi_top_band = max(rfi_top_band1,rfi_top_band2)
            if rfi_top_band>max_rfi:
                max_rfi = rfi_top_band
        returned_object = max_rfi
    else:
        returned_object = dm_time

    return returned_object

def load_filterbank(filterbank_name,dm=None,fullpol=False,burst_time=None):
    """ load_filterbank() adapted from K. Nimmo. bbpipe: https://github.com/KenzieNimmo/bbpipe.
    """
    fil = filterbank.FilterbankFile(filterbank_name)
    tsamp=fil.header['tsamp']

    burst_bin = int(burst_time/tsamp)

    #consider how much delay the DM would cause
    tdel=np.abs(4.149377593360996e-3*dm*((1./np.min(fil.frequencies/1000.)**2)-(1./np.max(fil.frequencies/1000.)**2))) #seconds
    bt=250e-3

    if int(2*tdel/tsamp) < 60000:
        t_duration = 60000
    elif int(2*tdel/tsamp) > 60000:
        t_duration = int(1.5*tdel/tsamp)

    if (burst_bin-int(250e-3/tsamp)+int(250e-3/tsamp) < fil.nspec) & (burst_bin-int(250e-3/tsamp) >= 0):
        spec = fil.get_spectra(burst_bin-int(250e-3/tsamp),t_duration)
        begbin=burst_bin-int(250e-3/tsamp)

    elif burst_bin-int(250e-3/tsamp) < 0:
        spec = fil.get_spectra(0,t_duration)
        bt = burst_bin*tsamp
        begbin=0
    else:
        dur = fil.nspec - (burst_bin-int(250e-3/tsamp))
        spec = fil.get_spectra(burst_bin-int(250e-3/tsamp),dur)
        begbin=burst_bin-int(250e-3/tsamp)

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

    return arr, extent, tsamp, begbin, arr_not_dedispersed, bt

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
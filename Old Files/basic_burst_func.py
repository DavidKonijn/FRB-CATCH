import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import burst_2d_Gaussian as fitter

from scipy.signal import correlate2d
from tqdm import trange
from scipy.optimize import curve_fit
from scipy import signal
from ACF_funcs import lorentz, autocorr
from lmfit import minimize, Parameters, fit_report, Model
from load_file import load_filterbank

def loaddata(filename, t_burst, DM=0, maskfile=None, window=100):
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
    ds,dsoff,extent,tsamp,begbin=load_filterbank(filename,dm=DM,fullpol=False,burst_time=t_burst)

    StokesI_ds = np.zeros_like(ds)
    StokesI_off = np.zeros_like(dsoff)
    #removing bandpass

    for fr in trange(ds.shape[0]):
        StokesI_ds[fr,:]=convert_SN(ds[fr,:],dsoff[fr,:])
        StokesI_off[fr,:]=convert_SN(dsoff[fr,:],dsoff[fr,:])

    # frequency resolution
    freqres=(extent[3]-extent[2])/ds.shape[0]
    # frequency array
    frequencies = np.linspace(extent[2],extent[3],ds.shape[0])

    if maskfile!=None:
        maskchans=np.loadtxt(maskfile,dtype='int')
        maskchans = [StokesI_ds.shape[0]-1-x for x in maskchans]
        StokesI_ds[maskchans,:]=0
        StokesI_off[maskchans,:]=0

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

    return StokesI_ds, StokesI_off, tsamp, freqres, begbin, frequencies

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

def convert_SN(burst_prof, off_prof):
    burst_prof-=np.mean(off_prof)
    off_prof-=np.mean(off_prof)
    burst_prof/=np.std(off_prof)
    return burst_prof

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def twoD_Gaussian(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta):
    (x,y) = x_data_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def twoD_Gaussian_fit(params,x_data_tuple,data):
    amplitude=params['amplitude']
    xo=params['xo']
    yo=params['yo']
    sigma_x = params['sigma_x']
    sigma_y = params['sigma_y']
    theta = params['theta']

    fit=twoD_Gaussian(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta)

    resid = data.ravel()-fit
    return resid

def downsamp(ds,tdown=1,fdown=1):
    tdown=int(tdown)
    fdown=int(fdown)

    if fdown!=1:
        ds=ds.reshape(ds.shape[0]//fdown, fdown,ds.shape[-1]).sum(axis=1)
        # ds_reshaped=ds.reshape(ds.shape[0]//fdown, fdown,ds.shape[-1])
        # ds=np.ma.masked_array(ds_reshaped.data.sum(axis=1), ds_reshaped.mask.any(axis=1))

    if tdown!=1:
        ds=ds.reshape(ds.shape[0], ds.shape[-1]//tdown, tdown).sum(axis=2)
    return ds

def window(ds,window,tsamp,begintime):
    """
    chop out a window around the burst (peak of the profile)
    """
    profile = np.mean(ds,axis=0)

    if begintime!=0:
        begin_b=int(0.1/tsamp)-int(window*2/(1000*tsamp))
        end_b=int(0.1/tsamp)+int(window*2/(1000*tsamp))
        profchop=profile[begin_b:end_b]
        begin=np.argmax(profchop)+begin_b - int(window/(1000*tsamp))
        end=begin + 2*int(window/(1000*tsamp))
        if np.max(profile)!=np.max(np.mean(ds[:,begin:end],axis=0)):
            begin=np.argmax(profile)-int(window/(1000*tsamp))
            end=np.argmax(profile)+int(window/(1000*tsamp))
    else:
        begin=np.argmax(profile)-int(window/(1000*tsamp))
        end=np.argmax(profile)+int(window/(1000*tsamp))
    burstds=ds[:,begin:end]
    return burstds,begin

def plot_ds(burstid, ds, outdir, name):
    plt.figure(figsize=(12,8))
    plt.imshow(ds, aspect='auto')
    plt.xlabel('Time', fontsize = 14)
    plt.ylabel('Observed Frequency', fontsize = 14)
    plt.savefig(outdir+'/B%s_'%burstid + name +'_Dynamic_Spectrum.pdf',format='pdf',dpi=300)

def twod_init(burst, dynspec, dynspec_off, begin_bin, tres, fres, freqs, downsample = False, outdir='./'):

    twidth,twidtherr,fwidth,fwidtherr,theta,thetaerr = twodacf(burst, dynspec, begin_bin, tres, fres, outdir=outdir)

    if downsample:
        tdown=8
        fdown=1

        dynspec_orig = dynspec.copy()
        dynspec_off_orig = dynspec_off.copy()
        freqs_orig=freqs.copy()
        tres_orig=tres

        while dynspec_orig.shape[1]%tdown !=0:
            dynspec_orig=dynspec_orig[:,:-1]
        while dynspec_off_orig.shape[1]%tdown !=0:
            dynspec_off_orig=dynspec_off_orig[:,:-1]

        dynspec=downsamp(dynspec_orig,tdown=tdown,fdown=fdown)
        dynspec_off=downsamp(dynspec_off_orig,tdown=tdown,fdown=fdown)

        min_f=freqs_orig.min() - fres/2
        max_f = freqs_orig.max() + fres/2
        new_res = (max_f-min_f)/(dynspec.shape[0])
        fres=new_res
        freqs=np.linspace((min_f+fres/2),(max_f-fres/2),dynspec.shape[0])
        tres=tres_orig*tdown

        plot_ds(burst, dynspec, outdir, 'Downsampled_'+str(tdown)+'x')

    return dynspec, dynspec_off, begin_bin, tres, fres, freqs, twidth, twidtherr, fwidth, fwidtherr, theta, thetaerr

def twodacf(burst, dynspec, begin_bin, timeres, freqres, outdir='./'):
    """
    Performs a 2D ACF on the burst dynamic spectrum.
    Fits Gaussians to the broad shape, to measure the burst time and frequency extent.
    Inputs:
    - burst is the burst identifier/name
     - ds is a numpy array containing the Stokes I dynamic spectrum of the burst
     - timeres and freqres are the time and frequency resolution of the dynamic spectrum in seconds and MHz respectively.
     - acf_load is a numpy file containing a previously calculated 2D ACF of the dynamic spectrum ds, if it exists. Default is to calculate the ACF.
     - if you want to save the ACF as a numpy array, save=True. Default not to save.
     - if you want diagnostic plots plotted to screen, set plot=True, else it will save the plots to the output directory, outdir.
    """

    #chop a window around the burst
    ds, beg_sm = window(dynspec,5.0,timeres,begin_bin)

    ACF = correlate2d(ds, ds)

    ACF/=np.max(ACF)
    ACFmasked = np.ma.masked_where(ACF==np.max(ACF),ACF) # mask the zero-lag spike

    ACFtime = np.sum(ACF,axis=0)
    ACFfreq = np.sum(ACF,axis=1)
    ACFtime = np.ma.masked_where(ACFtime==np.max(ACFtime),ACFtime)
    ACFfreq = np.ma.masked_where(ACFfreq==np.max(ACFfreq),ACFfreq)

    #make the time and frequency axes
    time_one = np.arange(1,ds.shape[1],1)*timeres*1000 #ms
    times = np.concatenate((-time_one[::-1],np.concatenate(([0],time_one))))
    freq_one = np.arange(1,ds.shape[0],1)*freqres
    freqs = np.concatenate((-freq_one[::-1],np.concatenate(([0],freq_one))))

    #1D Gaussian fitting to ACFtime and ACF freq
    try:
        poptt, pcovt = curve_fit(gaus, times, ACFtime, p0=[1,0,np.max(times)])
        poptf, pcovf = curve_fit(gaus, freqs, ACFfreq, p0=[1,0,np.max(freqs)])

        if np.median(pcovt) > 1e8:
            print("Could not do basic fitting due to infinite covariance")
            return 0,0,0,0,0,0
    except:
        print("Could not do basic fitting")
        return 0,0,0,0,0,0

    #2D Gaussian fitting
    timesh, freqs_m = np.meshgrid(times, freqs)
    timesh = timesh.astype('float64')
    freqs_m = freqs_m.astype('float64')

    #defining the parameters
    params = Parameters()
    params.add('amplitude', value=1)
    params.add('xo',value=0,vary=False)
    params.add('yo',value=0,vary=False)
    params.add('sigma_x',value=int(poptt[2]),min=poptt[2]-0.5*poptt[2], max=poptt[2]+0.5*poptt[2])
    params.add('sigma_y',value=int(poptf[2]),min=poptf[2]-0.5*poptf[2], max=poptf[2]+0.5*poptf[2])
    params.add('theta',value=0)

    out = minimize(twoD_Gaussian_fit, params, kws={"x_data_tuple": (timesh,freqs_m), "data": ACFmasked})
    print("Times (x) are in milliseconds and Frequencies (y) are in MHz")
    print(fit_report(out))

    data_fitted = twoD_Gaussian((timesh, freqs_m), out.params['amplitude'],out.params['xo'],out.params['yo'],out.params['sigma_x'],out.params['sigma_y'],out.params['theta'])
    data_fitted=data_fitted.reshape(len(freqs),len(times))

    #residuals
    ACFtimeresid = ACFtime-np.sum(data_fitted,axis=0)
    ACFfreqresid = ACFfreq-np.sum(data_fitted,axis=1)

    #plot
    fig = plt.figure(figsize=(8, 8))
    rows=3
    cols=3
    widths = [3, 1,1]
    heights = [1,1,3]
    gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

    cmap = plt.cm.gist_yarg

    ax1 = fig.add_subplot(gs[0,0]) # Time ACF
    ax1.plot(times,ACFtime,color='k')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.set_xlim(times[0],times[-1])
    ax1.plot(times,np.sum(data_fitted,axis=0),color='purple')

    ax2 = fig.add_subplot(gs[1,0],sharex=ax1) # Time ACF residuals
    ax2.plot(times,ACFtimeresid,color='k')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_xlim(times[0],times[-1])

    ax3 = fig.add_subplot(gs[2,0],sharex=ax2) # 2D ACF
    T,F=np.meshgrid(times, freqs)
    ax3.imshow(ACFmasked,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times[0],times[-1],freqs[0],freqs[-1]))
    ax3.contour(T,F,data_fitted,4, colors='r', linewidths=.5)
    ax3.set_ylabel('Freq lag [MHz]')
    ax3.set_xlabel('Time lag [ms]')

    ax4 = fig.add_subplot(gs[2,1],sharey=ax3) #Freq ACF residuals
    ax4.plot(ACFfreqresid,freqs,color='k')
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylim(freqs[0],freqs[-1])
    #ax4.plot(lorentz(freqs,result_freq.params['gamma'],result_freq.params['y0'],result_freq.params['c']),freqs,color='orange')

    ax5 = fig.add_subplot(gs[2,2],sharey=ax4) #Freq ACF
    ax5.plot(ACFfreq,freqs,color='k')
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set_ylim(freqs[0],freqs[-1])
    ax5.plot(np.sum(data_fitted,axis=1),freqs,color='purple')

    plt.savefig(str(outdir)+'/B%s_2d_acf_burst.pdf'%burst,dpi=300,format='pdf')

    #return sigma_time, sigma_time_error, sigma_frequency, sigma_frequency_error, drift theta
    return np.abs(out.params['sigma_x'].value),out.params['sigma_x'].stderr, np.abs(out.params['sigma_y'].value),out.params['sigma_y'].stderr,out.params['theta'].value,out.params['theta'].stderr

def fit_gaus(burstid, dynspec, begin_bin, frequencies,timeres,twidth_guess,fwidth_guess,outdir='./'):
    """
    Fits a 2D gaussian to the burst dynamic spectrum
    Inputs:
    - burstid: burst identifier
    - ds: dynamic spectrum of the burst
    - frequencies: array of frequencies matching the y-axis of the dynamic spectrum
    - tsamp: sampling time of the data
    - twidth_guess and fwidth_guess are the time [ms] and frequency [MHz] guesses for the burst widths.
    """
    #if you want to plot on screen then let the user identify the subbursts to fit
    #else just use the peak of the dynamic spectrum as the initial guess

    ds, beg_sm = window(dynspec,30,timeres,begin_bin)

    time_guesses = np.array([np.where(ds==np.max(ds))[1][0]])
    freq_guesses = np.array([np.where(ds==np.max(ds))[0][0]])
    amp_guesses = np.array([np.max(ds)])

    print("Frequency start should be around:", frequencies[-1])
    print("Max freq locatoin is freq_guesses", freq_guesses)

    # Get the times at the pixel centers in ms.
    times = (np.arange(ds.shape[1]) * timeres + timeres/2) * 1e3
    time_guesses = time_guesses*timeres*1e3
    freq_guesses = frequencies[-1] - np.array(freq_guesses, np.float64) * 4
    n_sbs = len(time_guesses)
    freq_std_guess = [fwidth_guess] * n_sbs
    t_std_guess = [twidth_guess/n_sbs] * n_sbs

    print(freq_guesses, freq_std_guess, t_std_guess)
    model = fitter.gen_Gauss2D_model(time_guesses, amp_guesses, f0=freq_guesses,bw=freq_std_guess, dt=t_std_guess, verbose=True)

    bestfit, fitLM = fitter.fit_Gauss2D_model(ds, times, frequencies, model)
    bestfit_params, bestfit_errors, corr_fig = fitter.report_Gauss_parameters(bestfit,fitLM,verbose=True)

    fig, res_fig = fitter.plot_burst_windows(times, frequencies,ds, bestfit, ncontour=8,res_plot=True)  # diagnostic plots
    fig.savefig(outdir+'/B%s_gausfit.pdf'%burstid)
    # res_fig.savefig(outdir+'/B%s_gausfit_residuals.pdf'%burstid)

    return bestfit_params, bestfit_errors

def compute_fluence(burstid, dynspec, begin_bin, tres, dsoff, t_width, f_width, fres, freqs, SEFD, gausfit, outdir='./'):
    """
    Converts burst profile to physical units
    Inputs:
    - ds is the dynamic spectrum,dsoff is the dynamic spectrum of off burst data
    - tcent, fcent are the centre time and frequency of the burst from the filterbank in ms and MHz, respectively
    - twidth, fwidth are the 1sigma width of the burst in time and frequency in ms and MHz, respectively.
    - tres and fres are the time resolution and frequency resolution in seconds and MHz, respectively.
    - freqs is the array of frequencies matching the y-axis of ds
    - start_time is the time in milliseconds from the start of the filterbank where the dynamic spectrum (ds) begins
    - SEFD is the system equivalent flux density of your telescope during your observation
    """

    ds, beg_sm = window(dynspec,30,tres,begin_bin)

    begintime=begin_bin*tres
    beg_window=begintime + beg_sm*tres
    start_time=beg_window*1000

    tcent = gausfit[0][1]+(beg_window*1000)
    fcent = gausfit[0][2]

    tburst = tcent - start_time
    tburst /= (tres*1000)
    tburst = int(tburst)
    fburst = int((fcent-freqs.min())/fres)

    begin_t=int(tburst-((2*t_width)/ (tres*1000)))
    end_t=int(tburst+((2*t_width)/ (tres*1000)))
    begin_f=int(fburst-((2*f_width)/ (fres)))
    end_f=int(fburst+((2*f_width)/ (fres)))

    if begin_f < 0:
        begin_f=0
    if end_f >= ds.shape[0] or end_f < 0:
        end_f=ds.shape[0]-1

    burst_ds = ds[begin_f:end_f, begin_t:end_t]
    off = dsoff[begin_f:end_f,100:100+(end_t-begin_t)]

    profile_burst = np.mean(burst_ds,axis=0)
    profile_off = np.mean(off,axis=0)
    profile_full = np.mean(ds[begin_f:end_f,:],axis=0)

    profile_burst=convert_SN(profile_burst, profile_off)
    profile_full=convert_SN(profile_full, profile_off)
    profile_off=convert_SN(profile_off, profile_off)

    bw = (end_f-begin_f)*fres
    profile_burst_flux=profile_burst*radiometer(tres*1000,bw,2,SEFD)
    fluence = np.sum(profile_burst_flux*tres*1000)

    width = (end_t - begin_t)

    print("S/N of burst B%s is "%(burstid)+str(np.sum(profile_burst)/np.sqrt(width)))
    print("Peak S/N of burst B%s is "%(burstid)+str(np.max(profile_burst)))
    print("Peak flux density of burst B%s is "%(burstid)+str(np.max(profile_burst_flux))+" Jy")
    print("Fluence of burst B%s is "%(burstid)+str(fluence)+" Jy ms")

    #return S/N, peak S/N, peak flux density, fluence
    return np.sum(profile_burst)/np.sqrt(width), np.max(profile_burst), np.max(profile_burst_flux), fluence
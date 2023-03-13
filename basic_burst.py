from basic_burst_func import *
import time
import numpy as np

start = time.time()

fil_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_1x5_8bit/lilo_1x5_8bit.fil'
mask_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_1x5_8bit/lilo_1x5_8bit_badchans.txt'
outdir = '/home/konijn/basicburst'

bursts = [1]
burst_time = [53.5541]
DM = 222.8890

# fil_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_2x3_8bit/lilo_2x3_8bit.fil'
# mask_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_2x3_8bit/lilo_2x3_8bit_badchans.txt'
# outdir = '/home/konijn/basicburst'

# bursts = [1]
# burst_time = [57.1856]
# DM = 224

# fil_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_2x4_8bit/lilo_2x4_8bit.fil'
# mask_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_2x4_8bit/lilo_2x4_8bit_badchans.txt'
# outdir = '/home/konijn/basicburst'

# bursts = [1]
# burst_time = [59.4621]
# DM = 218

# fil_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_4x2_8bit/lilo_4x2_8bit.fil'
# mask_file = '/data/konijn/burst_extraction/RNarwhal/59891/lilo_4x2_8bit/lilo_4x2_8bit_badchans.txt'
# outdir = '/home/konijn/basicburst'

# bursts = [1]
# burst_time = [47.5044]
# DM = 218

for bn, burst in enumerate(bursts):
    print('## Load the data ##')
    dynspec,dynspec_off,tres,fres,begin_bin,freqs=loaddata(fil_file, burst_time[bn], DM=DM, maskfile=mask_file)

    print("## Plot and save the Dynamic Spectrum ##")
    plot_ds(burst, dynspec, outdir, 'Simple')

    np.savetxt("dynspec_dimdim_burst.csv", dynspec, delimiter=",")

    print("## Calculate the 2D ACF ##")
    dynspec, dynspec_off, begin_bin, tres, fres, freqs, twidth, twidtherr, fwidth, fwidtherr, theta, thetaerr = twod_init(burst, dynspec, dynspec_off, begin_bin, tres, fres, freqs,  downsample = False, outdir=outdir)

    print("Estimated twidth and fwitdh:", twidth, fwidth)

    print("## Performing the 2D Gaussian fit analysis on the Dynamic Spectrum ##")
    gausfit, gausfit_errors = fit_gaus(burst, dynspec, begin_bin, freqs, tres, twidth, fwidth, outdir=outdir)

    print("## Performing fluence calculations for burst B%s ##"%burst)
    sn, peak_sn, peak_flux, fluence = compute_fluence(burst, dynspec, begin_bin, tres, dynspec_off, twidth, fwidth, fres, freqs, 25, gausfit, outdir=outdir)

    print(sn)

#     df = pd.DataFrame(columns=['Fil_file', 'Mask_file', 'Burst_time', 'Time_downsample', 'Time_width', 'Time_width_error','Freq_width', 'Freq_width_error', 'Theta','Theta_err','Scint_bw','Scint_bw_error','ncomp','TOA','FOA', 'S/N', 'Peak_S/N', 'Peak_flux', 'Fluence', 'Eiso', 'Espec', 'Lspec','ACF/gaus'])
#     df.loc[burst,'Fil_file']=fil_file
#     df.loc[burst,'Mask_file']=mask_file
#     df.loc[burst,'Burst_time']=burst_time[bn]
#     df.loc[burst,'Time_width']=twidth
#     df.loc[burst,'Time_width_error']=twidtherr
#     df.loc[burst,'Freq_width']=fwidth
#     df.loc[burst,'Freq_width_error']=fwidtherr
#     df.loc[burst,'Theta']=theta
#     df.loc[burst,'Theta_err']=thetaerr
#     df.loc[burst,'S/N']=sn
#     df.loc[burst,'Peak_S/N']=peak_sn
#     df.loc[burst,'Peak_flux']=peak_flux
#     df.loc[burst,'Fluence']=fluence

# df.to_csv(outdir + '/B%s_bb_analysis.csv'%burst)

end = time.time()

print("## Total burst analysis time:", np.round(end-start, 2), "s ##")
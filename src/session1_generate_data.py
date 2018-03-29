import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=18, usetex=True)
import time
from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap


#-------------------------------------------------------------------------------------------------
# I'm defining a function here that I use throughout this notebook 
# to generate mock data that we extract information from. 
# Use this function if you would like to generate alternative versions of the data, 
# eg. by changing the strength of the signal (SNR) or the number of exoplanets (num_exoplanets)
#-------------------------------------------------------------------------------------------------

def generate_fake_exoplanet_data(obs_time = 20, num_exoplanets = 3, size_timeseries = 500, star_lum = 10, 
                                 SNR = 0.0001, radii_ratios_scale = 0.1, error_mag = 0.002,
                                 seed_val = 2, box = False, verbose = True):

    np.random.seed(seed_val)    

    if box == True:
        radii_ratios_scale = radii_ratios_scale * 2

    exoplanet_radii_ratios = np.abs(np.random.normal(size=num_exoplanets)*radii_ratios_scale)
    exoplanet_periods = np.random.random(size=num_exoplanets)*obs_time
    exoplanet_start_locations = np.random.random(size=num_exoplanets)*obs_time

    for i in range(num_exoplanets):
        while (exoplanet_start_locations[i] > exoplanet_periods[i]):
            exoplanet_start_locations[i] = exoplanet_start_locations[i] - exoplanet_periods[i]

    timeseries = np.linspace(0,obs_time,size_timeseries)

    lum_time_base = np.ones_like(timeseries)*star_lum
    for i in range(num_exoplanets):
        drop_fac = exoplanet_radii_ratios[i]**2
        time_loc = exoplanet_start_locations[i]
        while (time_loc < obs_time):
            lum_time_base = lum_time_base - drop_fac*np.exp(-(timeseries-time_loc)**2/(2*np.pi*10*(drop_fac**2)))
            time_loc = time_loc + exoplanet_periods[i]

    if box == True: 
        mask = (lum_time_base < (np.amin(lum_time_base) + (np.amax(lum_time_base) - np.amin(lum_time_base))*0.4))
        lum_time_base[mask] = (np.amin(lum_time_base) + (np.amax(lum_time_base) - np.amin(lum_time_base))*0.4)


    lum_time_noisy = lum_time_base*(1+np.random.normal(size=size_timeseries)*SNR)
    lum_time_err = (lum_time_noisy)*(0.05*np.random.random(size=size_timeseries)+0.05)*error_mag

    if verbose == True:
        
        print('Periods: ', exoplanet_periods)
        
        plt.figure(figsize=(12,6))
        plt.plot(timeseries,lum_time_base,'k-',label='underlying signal')
        plt.errorbar(timeseries,lum_time_noisy,yerr = lum_time_err,lw=0,elinewidth=2,
                     marker='.',label='noisy observed data',alpha=0.9)

        plt.legend(fontsize=18,bbox_to_anchor=(1.4,1),edgecolor='w')
        plt.xlabel('time [days]',fontsize=24)
        plt.ylabel('Luminosity',fontsize=24)
        plt.title('random seed value: '+str(seed_val),fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

    return lum_time_noisy, lum_time_base, lum_time_err, timeseries, exoplanet_periods



#-------------------------------------------------------------------------------------------------
# Here we make a simple polynomial model for signal drift and try to recover our true parameters after accounting for it.

# timeax:     array, times at which observations are taken
# noisy_data: array, measured luminosity data
# poly order: integer, generates a polynomial drift of input order
# scale:      float, scales the drift to some overall factor of the signal.
# rand_val:   integer, sets random seed for reproducibility.


def add_drift(timeax, noisy_data, poly_order = 2, scale = 0.02, rand_val = 1):
    
    np.random.seed(rand_val)
    drift_coeffs = np.random.normal(size=(poly_order+1))
    drift_signal = np.zeros_like(timeax)
    
    for i in range(poly_order+1):
        drift_signal = drift_signal + drift_coeffs[i]*(timeax**i)
        
    scaled_drift_signal = (drift_signal / np.amax(np.abs(drift_signal))) * scale
        
    drift_data = noisy_data + scaled_drift_signal
    
    return drift_data, drift_coeffs


#-----------------------------------------------------------------------------------------------------
# Adding a function to do Lomb-Scargle periodogram analysis

def lomb_scargle_analysis(timeax,noisy_data,uncerts,true_periods = np.array([0])):

    time_start = time.time()

    #------------------------------------------------------------
    # Generate Data
    #np.random.seed(0)
    #N = 30
    #P = 0.3

    #t = np.random.randint(100, size=N) + 0.3 + 0.4 * np.random.random(N)
    #y = 10 + np.sin(2 * np.pi * t / P)
    #dy = 0.5 + 0.5 * np.random.random(N)
    #y_obs = np.random.normal(y, dy)

    t = timeax+1
    y_obs = noisy_data
    dy = uncerts


    #------------------------------------------------------------
    # Compute periodogram
    period = 10 ** np.linspace(-1, 1.8, 10000)
    omega = 2 * np.pi / period
    PS = lomb_scargle(t, y_obs, dy, omega, generalized=True)

    time_periodogram = time.time()

    print('time taken to compute periodogram: %.3f'  %(time_periodogram-time_start))

    #------------------------------------------------------------
    # Plot the results

    #true_periods = np.array([ 8.40735604  6.60669642  4.09297268])

    fig = plt.figure(figsize=(15, 3.75*3))
    fig.subplots_adjust(left=0.1, right=0.9, hspace=0.25)

    # First panel: the data
    ax = fig.add_subplot(211)
    ax.errorbar(t, y_obs, dy, fmt='.k', lw=1, ecolor='gray')
    ax.set_xlabel('time (days)',fontsize=18)
    ax.set_ylabel('flux',fontsize=18)
    ax.set_xlim(np.amin(t)-1,np.amax(t)+1)

    # Second panel: the periodogram & significance levels
    ax1 = fig.add_subplot(212, xscale='log')
    ax1.plot(period, PS, '-', c='black', lw=1, zorder=1)
    if len(true_periods) > 0:
        for i in range(len(true_periods)):
            plt.plot([true_periods[i],true_periods[i]],[0,np.amax(PS)*1.2],'k--')
    #plt.plot([true_periods[1],true_periods[1]],[0,np.amax(PS)*1.2],'k--')
    #plt.plot([true_periods[2],true_periods[2]],[0,np.amax(PS)*1.2],'k--')
    #plt.show()
    #ax1.plot([period[0], period[-1]], [sig1, sig1], ':', c='black')
    #ax1.plot([period[0], period[-1]], [sig5, sig5], ':', c='black')

    #ax1.annotate("", (0.3, 0.65), (0.3, 0.85), ha='center',
    #             arrowprops=dict(arrowstyle='->'))

    ax1.set_xlim(period[0], period[-1])
    #ax1.set_ylim(-0.05, 0.85)
    ax1.set_ylim(-0.05, np.amax(PS)*1.2)

    ax1.set_xlabel(r'period (days)',fontsize=18)
    ax1.set_ylabel('power',fontsize=18)

    # Twin axis: label BIC on the right side
    ax2 = ax1.twinx()
    ax2.set_ylim(tuple(lomb_scargle_BIC(ax1.get_ylim(), y_obs, dy)))
    ax2.set_ylabel(r'$\Delta BIC$')

    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_locator(plt.LogLocator(10))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))

    plt.show()


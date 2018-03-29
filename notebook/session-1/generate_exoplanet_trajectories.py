import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=18, usetex=True)


#-------------------------------------------------------------------------------------------------
# I'm defining a function here that I use throughout this notebook 
# to generate mock data that we extract information from. 
# Use this function if you would like to generate alternative versions of the data, 
# eg. by changing the strength of the signal (SNR) or the number of exoplanets (num_exoplanets)
#-------------------------------------------------------------------------------------------------

def generate_fake_exoplanet_data(obs_time = 20, num_exoplanets = 3, size_timeseries = 500, star_lum = 10, 
                                 SNR = 0.0001, radii_ratios_scale = 0.1, error_mag = 0.002,
                                 seed_val = 2, verbose = True):

    np.random.seed(seed_val)    

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

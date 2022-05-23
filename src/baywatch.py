import numpy as np
import pandas as pd
from sklearn import mixture
import multiprocessing

from .helpfns import *

### random permute data and find PSD threshold as the (permute_cnt * Confidence)-th Maximum PSD value
#def get_psd_threshold(data, permute_cnt=20, C=0.95, sample_freq=1):
def baywatch_permute(data, permute_cnt=20, C=0.95, sample_freq=1):
    max_psd = []
    for i in range(permute_cnt):
        permuted_data = np.random.permutation(data)
        _, t_psd = compute_psd(permuted_data, sample_freq)
        max_power = np.max(t_psd)
        max_psd.append(max_power)
        
    rank = int(C * permute_cnt) - 1
    max_psd.sort()
    psd_threshold = max_psd[rank]
    return psd_threshold


#### pvalue pruning 
def pvalue_pruning(tsintervals, potential_freqs, alpha=0.05):
    res = []
    for period in potential_freqs:
        _, pvalue = stats.ttest_1samp(tsintervals, period)       
        if pvalue > alpha or np.isnan(pvalue):
            res.append(period)
    return res

def gmm_fitting(tsintervals):
    bic = []
    n_components_range = range(1, 4)
    X = tsintervals.reshape(-1,1)
    lowest_bic = np.infty
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,)
        gmm.fit(X)
        bic.append(gmm.bic(X))

        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    return best_gmm.means_.flatten()


def baywatch_method(signals):
    """
    implementation of periodicity detection algorithm in ''Baywatch: robust beaconing detection to identify infected hosts in large-scale enterprise networks''
    
    Returns: 
    array: list of detected periods (empty if the method does not report specific detected periods)
    bool: True if the signal is periodic else False
    """
        
    periods = []
    detected = False
    
    freq, psd = compute_psd(signals)
    psd_threshold = baywatch_permute(signals)
    potential_pers = get_potential_periods(freq, psd, psd_threshold)
    
    # if no valid periodicity
    if len(potential_pers) == 0:
        return periods, detected

    ts_intervals = get_ts_intervals(signals)
    min_ts = get_min_tsinterval(ts_intervals) 
    high_freq_periods = high_freq_pruning(potential_pers, min_ts)
    
    # if no valid periodicity
    if len(high_freq_periods) == 0:
        return periods, detected
    
    ### pvalue pruning
    pvalue_pruned_periods = pvalue_pruning(ts_intervals, high_freq_periods)
    # if no valid periodicity
    if len(pvalue_pruned_periods) == 0:
        return periods, detected
    
    ### gmm fitting
    gmm_pers = gmm_fitting(ts_intervals)
    # if no valid periodicity
    if len(gmm_pers) == 0:
        return periods, detected
    
    ## acf verification
    acf_peaks = get_autocorr_peaks(signals)
    periods = acf_filtered_periodicity(gmm_pers, acf_peaks)
    if len(periods) > 0:
        detected = True
        
    return periods, detected

    
def baywatch_wrap(df):
    """
    wrap for data frame processing
    """
    df['periods'], df['detected'] = zip(*df["tdf"].apply(baywatch_method))
    return df

def mltproc_baywatch_wrap(df, maxproc = 16):
    """
    wrap func for multiprocessing
    """    
    pool = multiprocessing.Pool(processes = maxproc)    
    df_lst = np.array_split(df, maxproc)
    res = pool.map(baywatch_wrap, df_lst)
    resdf = pd.concat(res)
    return resdf


    """
    df = df.loc[df['ts_cnt'] > 2]
    if df.shape[0] == 0:
        return df[["tdf", "ts_cnt"]]

    ## find periodicity hints
    df['freq'], df['psd'] = zip(*df["tdf"].apply(compute_psd))
    df['psd_threshold'] = df["tdf"].apply(baywatch_permute)
    df["potential_periods"] = df.apply(lambda x: get_potential_periods(x['freq'], x['psd'], x['psd_threshold']), axis=1)
    
    ### filter entries that do not have any potential periods
    df = df.loc[df["potential_periods"].map(len) > 0 ]
   
    if df.shape[0] == 0:
        return df[["tdf", "ts_cnt"]]
    
    ### filter periods that are high freq noise
    df["ts_intervals"] = df['tdf'].apply(lambda x: get_ts_intervals(x))
    df = df.loc[df["ts_intervals"].map(len) > 0]
    df["min_tsinterval"] = df["ts_intervals"].apply(lambda x: get_min_tsinterval(x))
    df["high_freq_pruned"] = df.apply(lambda x: high_freq_pruning(x['potential_periods'], x['min_tsinterval']), axis=1)
    df = df.loc[df["high_freq_pruned"].map(len) > 0]
    
    if df.shape[0] == 0:
        return df[["tdf", "ts_cnt"]]  
    
    ### pvalue pruning
    df['pvalue_pruned_periods'] = df.apply(lambda x: pvalue_pruning(x['ts_intervals'], x['high_freq_pruned']), axis=1)
    df = df.loc[df['pvalue_pruned_periods'].map(len)>0]
    if df.shape[0] == 0:
        return df[["tdf", "ts_cnt"]]
    
    ### gmm fitting
    df["gmm_pers"] = df.apply(lambda x: gmm_fitting(x['ts_intervals']), axis=1)
    df = df.loc[df['gmm_pers'].map(len)>0]
    if df.shape[0] == 0:
        return df[["tdf", "ts_cnt"]]
    
    ## acf verification
    df['autocorr_peaks'] = df.apply(lambda x: get_autocorr_peaks(x['tdf']), axis=1)
    df['periods'] = df.apply(lambda x: acf_filtered_periodicity(x['gmm_pers'], x['autocorr_peaks']), axis=1)
    
    ## filtere df
    df = df.loc[df["periods"].map(len)>0]
    df["detected"] = True
    
    df = df[["tdf", "ts_cnt", "periods", "detected"]]
    return df"""
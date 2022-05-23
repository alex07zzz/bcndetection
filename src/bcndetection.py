import numpy as np
import pandas as pd
import emd
import multiprocessing

from .helpfns import *



### random permute data and find PSD threshold as the (permute_cnt * Confidence)-th Maximum PSD value
def bcn_permute(data, permute_cnt=100, C=0.95, sample_freq=1):
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


def emd_compose(signals):
    imf_opts = {'sd_thresh': 0.05}
    try: 
        emdsig = emd.sift.sift(signals, imf_opts=imf_opts, max_imfs=2)[:,0]
        return emdsig
    except:
        return signals


### filter potential periods using autocorr peaks
### Larger threshold here due to the fact that emd decomposition will shif the signals
def bcn_filtering(potential_periods, autocorr_peaks, threshold=2):
    true_period = []
    if len(autocorr_peaks) == 0 or len(potential_periods) == 0:
        return true_period
    
    for period in potential_periods:
        if min(abs(autocorr_peaks - period)) <= threshold:
            true_period.append(period)
    return true_period
    
def bcndetection_method(signals):
    """
    implementation of our periodicity detection algorithm
    
    Returns: 
    array: list of detected periods (empty if the method does not report specific detected periods)
    bool: True if the signal is periodic else False
    """
    periods = []
    detected = False
    
    # decompose
    signals = emd_compose(signals)
    freq, psd = compute_psd(signals)
    psd_threshold = bcn_permute(signals)
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
    
    ## acf verification
    acf_peaks = get_autocorr_peaks(signals)
    periods = bcn_filtering(high_freq_periods, acf_peaks)
    if len(periods) > 0:
        detected = True
        
    return periods, detected

    
def bcndetection_wrap(df):
    """
    wrap for data frame processing
    """
    df['periods'], df['detected'] = zip(*df["tdf"].apply(bcndetection_method))
    return df

def mltproc_bcndetection_wrap(df, maxproc = 16):
    """
    wrap func for multiprocessing
    """    
    pool = multiprocessing.Pool(processes = maxproc)    
    df_lst = np.array_split(df, maxproc)
    res = pool.map(bcndetection_wrap, df_lst)
    resdf = pd.concat(res)
    return resdf


"""
def bcndetection_wrap(df, mute=True):
    
    ## filter ts_cnt > 2
    df = df.loc[df['ts_cnt'] > 2]
    
    if not mute: 
        print("total entries:", df.shape[0])
        
    if df.shape[0] == 0:
        return df
    
    ### decompose
    df["hht"], df["composed"] = zip(*df["tdf"].apply(emd_compose))
    df = df.drop(columns=["tdf"])
    df = df.rename(columns={"hht": "tdf"})

    ## find periodicity hints
    df['freq'], df['psd'] = zip(*df["tdf"].apply(compute_psd))
    df['psd_threshold'] = df["tdf"].apply(bcn_permute)
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
    
    df["psd_ratio"] = df.apply(lambda x: psd_ratio(x['psd'], x['psd_threshold']), axis=1)
    
    df['autocorr_peaks'] = df.apply(lambda x: get_autocorr_peaks(x['tdf']), axis=1)
    df['periods'] = df.apply(lambda x: acf_filtered_periodicity(x['high_freq_pruned'], x['autocorr_peaks']), axis=1)

    ### Filtered
    df = df.loc[df["periods"].map(len)>0]
    df["detected"] = True
    
    df = df[["tdf", "ts_cnt", "periods", "detected"]]
    return df"""
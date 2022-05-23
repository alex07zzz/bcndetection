import numpy as np
import random

def omit(rate):
    return random.random() <= rate

def gen_periodic_signal_insec(period, std, omit_rate=0, length=1440 * 60):
    base = np.zeros(length)
        
    for i in range(length):
        
        if i % period == 0:
            
            if std == 0:
                shift = 0
            else:
                shift = round(np.random.normal(0, std))
            
            if (i + shift) < 0 or (i+shift) >= length:
                continue
            else:
                ## if the signal is not omitted, create signal
                if not omit(omit_rate):
                    base[i + shift] += 1
    return base


def resample_sig(sig, samplerate = 60):
    # resample signal
    return np.array([sum(sig[i: i+samplerate]) for i in range(0, len(sig), samplerate)])

def gen_signal_df(period, std=0, omit_rate=0, count=100, length=1440, samplerate=60):
    sigl = []
    
    for i in range(count):
        sig = np.zeros(length)
        sig_1sec = gen_periodic_signal_insec(period=period, std=std, omit_rate=omit_rate, length=length*60)
        sig += resample_sig(sig_1sec, samplerate)
        sigl.append(sig)
    
    sigdf = pd.DataFrame()
    sigdf["tdf"] = sigl
    sigdf["ts_cnt"] = sigdf["tdf"].apply(lambda x: len(x[x>0]))
    return sigdf
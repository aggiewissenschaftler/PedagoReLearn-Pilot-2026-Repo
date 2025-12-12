from typing import List, Any
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

class EpisodeTimeseriesResult:
    def __init__(self):
        self.mastery: List[float] = []
        self.recency: List[float] = []
        self.i_type: List[int] = []
        self.rule_id: List[int] = []
        self.reward: List[float] = []

DATA: List[EpisodeTimeseriesResult] = []

def append_new_EpisodeResult():
    DATA.append(EpisodeTimeseriesResult())

def append_stepResult(step, obs, i_type, rule_id, reward):
    tsResult:EpisodeTimeseriesResult = DATA[-1]

    tsResult.mastery.append(np.mean(obs['mastery']))
    tsResult.recency.append(np.mean(obs['recency']))
    i_id = ['noop','teach','quiz','review'].index(i_type)
    tsResult.i_type.append(i_id)
    tsResult.rule_id.append(rule_id)
    tsResult.reward.append(reward)
    
def append_episodeResult(result: EpisodeTimeseriesResult):
    DATA.append(result)

def save_n_reset(dscr=''):
    fname = 'ts_results\\'+dscr+datetime.strftime(datetime.now(),"%d%m%Y-%H%M%S.pkl")
    with open(fname, 'wb') as f:
        pickle.dump(DATA,f)
    # print("Full Episodes results saved to "+fname)
    
    DATA.clear()

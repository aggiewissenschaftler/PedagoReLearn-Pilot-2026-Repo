import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pickle
from glob import glob
import uuid

from utils.data_util import EpisodeTimeseriesResult

def load_np_data(path):
    with open(path, 'rb') as f:
        fr = pickle.load(f) # full result

    w = len(fr)
    h = max([len(ep_rst.mastery) for ep_rst in fr])

    # mastery/recency/cum_reward/teach/quiz/review
    np_names = ['Mastery','Recency','Cum_reward','Teach','Quiz','Review']
    np_result = np.full((6,h,w), np.nan, dtype=float) 

    # mastery, recency, cum_reward
    ep_rst: EpisodeTimeseriesResult 
    for e, ep_rst in enumerate(fr):
        for i, arr in enumerate([ep_rst.mastery, ep_rst.recency, ep_rst.reward]):
            L = len(arr)
            if L > 0:
                np_result[i,:L, e] = arr
        
    # i_type need to be converted to teach/quiz/review
    window = np.ones(max(1,h//10), dtype=float)
    for e, ep_rst in enumerate(fr):
        i_type = np.array(ep_rst.i_type)
        if len(i_type) >= 1 and len(i_type) >= len(window):

            # if e == 10: print(i_type)

            for i_type_id in range(1,4):
                arr = np.convolve(i_type==i_type_id,window,mode='valid')

                # if e == 10: print(arr)

                L = len(arr)
                W = len(window)
                np_result[i_type_id+2,W-1:L+W-1,e] = arr
        
    return np_names, np_result


def load_n_plot_multipath(paths):
    # average multiple seed
    np_names, np_result = load_np_data(paths[0])
    np_val = np.nan_to_num(np_result, nan=0)
    np_cnt = (~np.isnan(np_result)).astype(float)

    for path in paths[1:]:
        _, res = load_np_data(path)
        res_val = np.nan_to_num(res, nan=0)
        res_cnt = (~np.isnan(res)).astype(float)
        if np_val.shape[1] >= res_val.shape[1]:
            np_val[:,:res_val.shape[1],:] += res_val
            np_cnt[:,:res_cnt.shape[1],:] += res_cnt
        else:
            res_val[:,:np_val.shape[1],:] += np_val
            res_cnt[:,:np_cnt.shape[1],:] += np_cnt
            np_val = res_val
            np_cnt = res_cnt
    
    np_val /= (np_cnt + (np_cnt == 0)*1)
    
    # alpha-blended heatmaps
    k, h, w = np_val.shape
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True)

    for idx, (ax, label) in enumerate(zip(axes.flat, np_names)):
        val = np_val[idx]
        cnt = np_cnt[idx]

        # normalizer
        vmin = np.min(val)
        vmax = np.max(val)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        
        norm_val = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = plt.get_cmap('viridis')
        rgba = cmap(norm_val(val))

        # alpha from counts: per-plot normalization
        cnt_min = np.min(cnt)
        cnt_max = np.max(cnt)
        if cnt_max == cnt_min:
            cnt_min, cnt_max = 0.0, max(1.0, cnt_max)
        norm_alpha = colors.Normalize(vmin=cnt_min, vmax=cnt_max, clip=True)
        alpha = norm_alpha(cnt)
        alpha[cnt <= 0] = 0.0
        rgba[..., 3] = alpha

        im = ax.imshow(
            rgba,
            origin="lower",
            aspect="auto",
            interpolation="nearest"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Step")
        ax.set_title(f'{label} by Episode and Step')

        # colorbar
        sm = cm.ScalarMappable(norm=norm_val, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=label)

    fig.suptitle(f'Results of {w} Episodes (alpha-blended over {len(paths)} seeds)', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig(f'fig_results/full_{uuid.uuid1()}.png')

def plot(path_re):
    paths = glob(path_re)
    if len(paths) == 0:
        return
    
    load_n_plot_multipath(paths)


if __name__ == '__main__':
    plot('ts_results\\sarsa*')
    plot('ts_results\\random*')
    plot('ts_results\\fixed*')
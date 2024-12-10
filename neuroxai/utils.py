# -*- coding:utf-8 -*-
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler


def topography(ch_importance, title=None, ax=None):
    bio_semi_montage = mne.channels.make_standard_montage('standard_1005')
    ch_importance = [(ch, i) for ch, i in ch_importance.items()]
    cn_list, ci_list, pos_list = [], [], []
    for cn, ci in ch_importance:
        cn_list.append(cn)
        ci_list.append(ci)
        pos_list.append(bio_semi_montage.get_positions()['ch_pos'][cn])

    fake_info = mne.create_info(ch_names=cn_list, sfreq=512., ch_types='eeg')
    data = np.array(ci_list).reshape(-1, 1)
    data = StandardScaler().fit_transform(data)
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(bio_semi_montage)

    if title:
        ax.set_title(title)
    im, cn = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, sensors=True,
                                  names=None, ch_type='eeg', show=False, axes=ax)
    return im, cn

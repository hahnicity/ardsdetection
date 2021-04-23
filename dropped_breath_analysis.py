import argparse

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('unframed', help='unframed dataset with ventmode and tor')
    parser.add_argument('--data-cls', help='path to the dataset object with dropped data information', default='results/data-cls-obj-with-dropped-data.pkl')
    args = parser.parse_args()

    uvt = pd.read_pickle(args.unframed)
    _, data_cls = pd.read_pickle(args.data_cls)
    uvt.loc[uvt.y == 2, 'y'] = 0
    uvt.loc[uvt.dta == 2, 'dta'] = 1

    vm_mapping = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
    # find total distribution of asynchronies and ventmodes
    all_async = [0, 0, 0]

    # vc, pc, ps, pav, cpap
    all_vm = [0, 0, 0, 0, 0]

    for k, v in uvt.ventmode.value_counts().items():
        all_vm[vm_mapping[k]] += v

    all_async[1] += uvt.bsa.sum()
    all_async[2] += uvt.dta.sum()
    all_async[0] += len(uvt) - (uvt.bsa.sum() + uvt.dta.sum())

    # first asynchronies
    # normal, bsa, dta
    single_drop_async = [0, 0, 0]

    # ventmodes
    # vc, pc, ps, pav, cpap
    single_drop_vm = [0, 0, 0, 0, 0]

    patho_to_async_drop_breaths = {0: {'dta': 0, 'bsa': 0}, 1: {'dta': 0, 'bsa': 0}}
    # work on singly dropped breaths first
    for patient, vals in data_cls.dropped_data.items():
        dropped_breaths = vals['nan_inf_dropping']['drop_vent_bns']
        if not dropped_breaths:
            continue
        b_data = uvt[(uvt.patient == patient) & (uvt.ventBN.isin(dropped_breaths))]
        bsa_count = b_data.bsa.sum()
        dta_count = b_data.dta.sum()
        single_drop_async[1] += bsa_count
        single_drop_async[2] += dta_count
        single_drop_async[0] += len(b_data) - (b_data.bsa.sum() + b_data.dta.sum())
        for k, v in b_data.ventmode.value_counts().items():
            single_drop_vm[vm_mapping[k]] += v
        pt_patho = uvt[uvt.patient == patient].y.iloc[0]
        patho_to_async_drop_breaths[pt_patho]['bsa'] += bsa_count
        patho_to_async_drop_breaths[pt_patho]['dta'] += dta_count

    # first asynchronies
    # normal, bsa, dta
    frame_drop_async = [0, 0, 0]

    # ventmodes
    # vc, pc, ps, pav, cpap
    frame_drop_vm = [0, 0, 0, 0, 0]

    # next onto frame breaths first
    for patient, vals in data_cls.dropped_data.items():
        dropped_breaths = vals['too_many_discontinuous_bns']['vent_bns']
        if not dropped_breaths:
            continue
        tmp = []
        for stack in dropped_breaths:
            tmp.extend(stack)
        dropped_breaths = tmp
        b_data = uvt[(uvt.patient == patient) & (uvt.ventBN.isin(dropped_breaths))]
        bsa_count = b_data.bsa.sum()
        dta_count = b_data.dta.sum()
        frame_drop_async[1] += b_data.bsa.sum()
        frame_drop_async[2] += b_data.dta.sum()
        frame_drop_async[0] += len(b_data) - (b_data.bsa.sum() + b_data.dta.sum())
        for k, v in b_data.ventmode.value_counts().items():
            frame_drop_vm[vm_mapping[k]] += v
        patho_to_async_drop_breaths[pt_patho]['bsa'] += bsa_count
        patho_to_async_drop_breaths[pt_patho]['dta'] += dta_count

    p_all_async = (np.array(all_async) / float(sum(all_async)) * 100).round(2)
    p_all_vm = (np.array(all_vm) / float(sum(all_vm)) * 100).round(2)
    p_sd_async = (np.array(single_drop_async) / float(sum(single_drop_async)) * 100).round(2)
    p_sd_vm = (np.array(single_drop_vm) / float(sum(single_drop_vm)) * 100).round(2)
    p_fd_async = (np.array(frame_drop_async) / float(sum(frame_drop_async)) * 100).round(2)
    p_fd_vm = (np.array(frame_drop_vm) / float(sum(frame_drop_vm)) * 100).round(2)
    print(p_all_vm, p_sd_vm, p_fd_vm)
    x = np.arange(len(frame_drop_vm))
    width = 0.2
    plt.bar(x-width, p_all_vm, width, label='proportion all breaths')
    plt.bar(x, p_sd_vm, width, label='proportion single breaths dropped')
    plt.bar(x+width, p_fd_vm, width, label='proportion frames dropped')
    plt.xticks(x, ['VC', 'PC', 'PS', 'PAV', 'CPAP'])
    plt.ylabel('% Percentage')
    plt.legend()
    plt.savefig('/home/greg/ardsresearch/percentage-dropped-by-vm.png', dpi=1200)
    plt.close()

    x = np.arange(len(p_all_async))
    plt.bar(x-width, p_all_async, width, label='proportion all breaths')
    plt.bar(x, p_sd_async, width, label='proportion single breaths dropped')
    plt.bar(x+width, p_fd_async, width, label='proportion frames dropped')
    plt.xticks(x, ['Normal', 'BSA', 'DTA'])
    plt.ylabel('% Percentage')
    plt.legend()
    plt.savefig('/home/greg/ardsresearch/percentage-dropped-by-async.png', dpi=1200)
    plt.close()

    # now need to show dta and bsa by pathophys, before and after filter
    # this next chunk calcs before
    percentages = []
    abs_val_counts = []
    for val, df in uvt.groupby('y'):
        percentages.append([val, df.bsa.value_counts(normalize=True)[1], df.dta.value_counts(normalize=True)[1]])
        abs_val_counts.append([val, df.bsa.value_counts()[1], df.dta.value_counts()[1]])
    bsa_pre = np.array([percentages[0][1], percentages[1][1]]) * 100
    dta_pre = np.array([percentages[0][2], percentages[1][2]]) * 100
    # calc post
    post_val_counts = [
        [0, abs_val_counts[0][1] - patho_to_async_drop_breaths[0]['bsa'], abs_val_counts[0][2] - patho_to_async_drop_breaths[0]['dta']],
        [1, abs_val_counts[1][1] - patho_to_async_drop_breaths[1]['bsa'], abs_val_counts[1][2] - patho_to_async_drop_breaths[1]['dta']]
    ]
    total_breaths_end = 19777 * 100
    bsa_post = np.array([post_val_counts[0][1], post_val_counts[1][1]]) / float(total_breaths_end) * 100
    dta_post = np.array([post_val_counts[0][2], post_val_counts[1][2]]) / float(total_breaths_end) * 100

    width = 0.4
    x = np.array([0, 1])
    plt.bar(x-width/2, bsa_post, width, label='BSA', color='tab:blue')
    plt.bar(x+width/2, dta_post, width, label='DTA', color='tab:orange')
    plt.ylabel('Percentage Breaths in Final Dataset')
    plt.xticks(x, ['Non-ARDS', 'ARDS'])
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.grid(True, which='major', alpha=0.5)
    plt.legend()
    plt.savefig('/home/greg/ardsresearch/percentage-async-by-patho.tif', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

import argparse

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('unframed', help='unframed dataset with ventmode and tor')
    parser.add_argument('--data-cls', help='path to the dataset object with dropped data information', default='results/data-cls-obj-with-dropped-data.pkl')
    args = parser.parse_args()

    uvt = pd.read_pickle(args.unframed)
    data_cls = pd.read_pickle(args.data_cls)

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

    # work on singly dropped breaths first
    for patient, vals in data_cls.dropped_data.items():
        dropped_breaths = vals['nan_inf_dropping']['drop_vent_bns']
        if not dropped_breaths:
            continue
        b_data = uvt[(uvt.patient == patient) & (uvt.ventBN.isin(dropped_breaths))]
        single_drop_async[1] += b_data.bsa.sum()
        single_drop_async[2] += b_data.dta.sum()
        single_drop_async[0] += len(b_data) - (b_data.bsa.sum() + b_data.dta.sum())
        for k, v in b_data.ventmode.value_counts().items():
            single_drop_vm[vm_mapping[k]] += v

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
        frame_drop_async[1] += b_data.bsa.sum()
        frame_drop_async[2] += b_data.dta.sum()
        frame_drop_async[0] += len(b_data) - (b_data.bsa.sum() + b_data.dta.sum())
        for k, v in b_data.ventmode.value_counts().items():
            frame_drop_vm[vm_mapping[k]] += v

    p_all_async = (np.array(all_async) / float(sum(all_async)) * 100).round(2)
    p_all_vm = (np.array(all_vm) / float(sum(all_vm)) * 100).round(2)
    p_sd_async = (np.array(single_drop_async) / float(sum(single_drop_async)) * 100).round(2)
    p_sd_vm = (np.array(single_drop_vm) / float(sum(single_drop_vm)) * 100).round(2)
    p_fd_async = (np.array(frame_drop_async) / float(sum(frame_drop_async)) * 100).round(2)
    p_fd_vm = (np.array(frame_drop_vm) / float(sum(frame_drop_vm)) * 100).round(2)

    x = np.arange(len(frame_drop_vm))
    width = 0.2
    plt.bar(x-width, p_all_vm, width, label='all')
    plt.bar(x, p_sd_vm, width, label='single drop')
    plt.bar(x+width, p_fd_vm, width, label='frame drop')
    plt.xticks(x, ['VC', 'PC', 'PS', 'PAV', 'CPAP'])
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()

    x = np.arange(len(p_all_async))
    plt.bar(x-width, p_all_async, width, label='all')
    plt.bar(x, p_sd_async, width, label='single drop')
    plt.bar(x+width, p_fd_async, width, label='frame drop')
    plt.xticks(x, ['Normal', 'BSA', 'DTA'])
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def correlation_mat(df, annot):
    feats = [col for col in df.columns if 'nanmedian' in col]
    renaming = {f: f.replace('nanmedian_', '').replace('_', ' ') for f in feats}
    df = df.rename(columns=renaming)
    new_feat_names = list(renaming.values())
    corr_mat = df[new_feat_names].corr()
    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(corr_mat, cmap=cmap, annot=annot, annot_kws={'fontsize': 6})
    plt.tight_layout()
    plt.savefig('img/feature_correlation_matrix.png', dpi=200)
    plt.close()


def stat_compliance_resistance_viz(df_before, df_filtered):
    feats = [col for col in df_before.columns if 'nanmedian' in col]
    renaming = {f: f.replace('nanmedian_', '').replace('_', ' ') for f in feats}
    df_before = df_before.rename(columns=renaming)
    df_filtered = df_filtered.rename(columns=renaming)
    figsize = (3*4, 3*2)

    for col, pretty_name in [('resist', 'Resistance'), ('stat compliance', 'Static Compliance')]:
        pl, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for i, (ax_lab, frame) in enumerate([('Before', df_before), ('After', df_filtered)]):
            for target, label in [(0, 'Other'), (1, 'ARDS'), (2, 'COPD')]:
                sns.distplot(frame[frame.y==target][col], label=label, ax=axes[i])
            axes[i].set_title(ax_lab, fontsize='xx-large')
            axes[i].set_xlabel(pretty_name)
            axes[i].grid(alpha=.3, lw=1.5)
            axes[i].legend(title='Pathophysiology')
        plt.savefig('img/{}-distplot-before-after-filtering.png'.format(col.replace(' ', '-')), dpi=200)
        plt.close()

    # Visualize outliers on static compliance distribution
    for col, pretty_name in [('stat compliance', 'Static Compliance')]:
        pl, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for i, (ax_lab, frame) in enumerate([('Before', df_before), ('After', df_filtered)]):
            for target, label in [(0, 'Other'), (1, 'ARDS'), (2, 'COPD')]:
                sns.distplot(frame[frame.y==target][col], label=label, ax=axes[i])
            axes[i].set_title(ax_lab, fontsize='xx-large')
            axes[i].set_xlabel(pretty_name)
            axes[i].grid(alpha=.3, lw=1.5)
            axes[i].set_xlim(75, axes[i].get_xlim()[-1])
            axes[i].set_ylim(0, .005)
            axes[i].legend(title='Pathophysiology')
        plt.savefig('img/{}-distplot-before-after-filtering-zoom-in.png'.format(col.replace(' ', '-')), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('df')
    parser.add_argument('--df-filtered', help='data frame of filtered data. is optional')
    parser.add_argument('--anno-vals', action='store_true', help='annotate values in correlation matrix')
    args = parser.parse_args()

    df = pd.read_pickle(args.df)
    correlation_mat(df, args.anno_vals)
    if args.df_filtered:
        df_filtered = pd.read_pickle(args.df_filtered)
        stat_compliance_resistance_viz(df, df_filtered)


if __name__ == '__main__':
    main()

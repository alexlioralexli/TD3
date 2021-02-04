import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import namedtuple


RunResult = namedtuple('RunResult', 'dataframe folder_path')

plot_config = json.load(open('plot_config.json'))


def plot_data(data, value='exploration/num steps total', insert_legend=False, only_legend=False, remove_ylabel=True, large=True, title=None, args=None):

    data = list(reversed(data))  # so that HiPPO is on top of everything else
    assert isinstance(data, list)

    ### data smoothing and aggregation
    new_data = []
    for result in data:
        d = result.dataframe
        average_return = d[value][-5:].mean()
        print(d['Condition'][0], average_return, result.folder_path)
        d[value] = d[value].rolling(5, min_periods=1, center=True).mean()  # [:length] # 20 for the last few exps
        if args.xlimit:
            d = d[d['Timestep'] <= args.xlimit]
        new_data.append(d)
        print(len(d))
    data = pd.concat(new_data, ignore_index=True, sort=True)

    ### figure style
    # sns.set(style="darkgrid", font_scale=1) # , rc={'figure.figsize':(10, 7.5)})
    if large:
        dpi=1000
        font_scale=2.5 # 2
        # font_scale=1.8  #for hyperparameter sensitivity
    else:
        dpi=1000
        font_scale=1.5
    # sns.set(font='serif')
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams['font.serif'] = "Times New Roman"
    sns.set_context(context="paper", font_scale=font_scale)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    sns.set_style('whitegrid', {'axes.grid': True, 'grid.linestyle': '--', 'axes.spines.bottom': True,
                                'axes.spines.left': True, 'axes.spines.right': False, 'axes.spines.top': False})

    BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, MAGENTA, GREY, TAN, CYAN = sns.color_palette()
    COLORS = {
        "Fourier MLP (ours)": BLUE,
        'MLP': ORANGE,
    }
    if only_legend:
        fig, ax = plt.subplots(dpi=dpi, figsize=(12.5, 5))
    else:
        fig, ax = plt.subplots(dpi=dpi, figsize=(6.4, 4.8))
    ax.spines['left'].set_color('0.1')
    ax.spines['bottom'].set_color('0.1')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6)) #(6,6)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
    # plt.ylim(-2.6, 4.35)  #todo: for ant finetuning
    if title is not None:
        plt.title(title)

    ### plotting lines
    # sns.lineplot(x="Iteration", y="AverageReturn", hue="Condition", data=data, ci='sd', linewidth=3.0, palette=COLORS) #todo: for hyperparameter sensitivityfix
    sns.lineplot(x='Timestep', y=value, hue="Condition", data=data, ci='sd', linewidth=2.5, palette=COLORS) # 68 for the right sd
    plt.xlabel('Steps')
    if remove_ylabel:
        plt.ylabel('')
    else:
        plt.ylabel("Average Return")

    ### legend
    handles, labels = ax.get_legend_handles_labels()
    # leg = ax.legend(handles=handles[1:], labels=labels[1:], loc='upper right')
    # leg = ax.legend(handles=handles[len(handles)//2+1:], labels=labels[len(handles)//2+1:], loc='lower right')
    leg = ax.legend(handles=handles[1:], labels=labels[1:], loc='lower right')  #todo: may have to fix
    if only_legend:
        leg = ax.legend(handles=handles[1:], labels=labels[1:], loc='upper center', ncol=5,
                        framealpha=1)  # when i want the colors
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    if not insert_legend:
        ax.get_legend().remove()  # when i don't want the legend inside

    plt.tight_layout(pad=0)

def get_exp_name(variant):
    if variant.get('network_class').get('$class') == "models.mlp.FourierMLP":
        return "Fourier MLP (ours)"
    elif variant.get('network_class').get('$class') == "models.mlp.MLP":
        return "MLP"
    raise RuntimeError

def get_datasets(fpath, filter=None, valid_exp_names=None):
    assert filter is not None
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'progress.csv' in files:
            param_path = None
            exp_name = dir
            if 'params.json' in files:
                param_path = open(os.path.join(root, 'params.json'))
            elif 'variant.json' in files:
                param_path = open(os.path.join(root, 'variant.json'))
            assert param_path
            if param_path is not None:
                variant = json.load(param_path)
                if variant is None:
                    continue
                if not filter(variant):
                    continue
                exp_name = get_exp_name(variant)
                print(exp_name)
                if valid_exp_names and exp_name not in valid_exp_names:  # remove experiments that we don't want to plot here
                    continue

            log_path = os.path.join(root, 'progress.csv')
            try:
                experiment_data = pd.read_csv(log_path)
                experiment_data.insert(len(experiment_data.columns), 'Unit', unit)
                experiment_data.insert(len(experiment_data.columns), 'Condition', exp_name)
                datasets.append(RunResult(experiment_data, root))
                unit += 1
            except pd.io.common.EmptyDataError:
                print("Empty Data Error")
    # todo: get rid of failed ones
    # mean_length = np.mean([len(dataset) for dataset in datasets])
    # filtered_datasets = [dataset for dataset in datasets if len(dataset) > 0.85 * mean_length]
    # print("remaining: {}, num removed: {}".format(len(filtered_datasets), len(datasets) - len(filtered_datasets)))
    filtered_datasets = datasets
    return filtered_datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='Eval returns', nargs='*')
    parser.add_argument('--baselines', '-bl', default=None, nargs='*')
    parser.add_argument('--title', '-t', type=str, default=None)
    parser.add_argument('--save_path', '-sp', type=str, default=None)
    parser.add_argument('--ylabel', '-y', action='store_true')
    parser.add_argument('--insertlegend', '-l', action='store_true')
    parser.add_argument('--valid_exp_names', nargs='*', help='Names of lines to plot')
    parser.add_argument('--xlimit', type=int, default=None, help='Limit on range of the xaxis: [0, xlimit')
    args = parser.parse_args()
    print("--------------------------------------------------------")
    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    # to filter out the wrong lat dim directories
    def default_filter(variant):
        if variant['env'] in plot_config.keys():
            answer_key = plot_config[variant['env']][variant['network_class']['$class']]
        else:
            answer_key = plot_config[variant['network_class']['$class']]
        for k, v in answer_key.items():
            if k not in variant['network_kwargs'] or variant['network_kwargs'][k] != v:
                return False
        return True

    filter = default_filter
    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            print(logdir)
            data += get_datasets(logdir, filter=filter, valid_exp_names=args.valid_exp_names)
    else:
        for logdir in args.logdir:
            print(logdir)
            data += get_datasets(logdir, filter=filter, valid_exp_names=args.valid_exp_names)
    data = sorted(data, key=lambda result: result.dataframe['Condition'][0])
    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]

    sns.set_context("paper")
    sns.set()
    for value in values:
        plot_data(data,
                  value=value,
                  remove_ylabel=not args.ylabel,
                  insert_legend=args.insertlegend,
                  large=bool(args.save_path),
                  title=args.title,
                  args=args,
                  only_legend=args.insertlegend and args.save_path is not None and 'legend' in args.save_path)
    if args.save_path is None:
        plt.show()
    else:
        plt.savefig(args.save_path)

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def verbose_print(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.head())


def random_color():
    return list(np.random.choice(range(256), size=3))


def pie_chart(dataset, column):
    col = dataset[column].value_counts()
    col_vals = col.values.tolist()
    col_types = col.axes[0].tolist()

    colors = [random_color() for i in range(len(col_types))]
    colors = ['#%02x%02x%02x' % tuple(x) for x in colors]

    fig = plt.figure(figsize=(12, 8))

    # Plot
    plt.title('Feature: {}'.format(column), fontsize=20)
    patches, texts, autotexts = plt.pie(col_vals, labels=col_types, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=150)
    for text, autotext in zip(texts, autotexts):
        text.set_fontsize(14)
        autotext.set_fontsize(14)

    plt.axis('equal')
    plt.show()


def autolabel(ax, rects, fontsize=14):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=fontsize)

def binary_bar_chart(dataset, column):
    above_list = []
    below_list = []

    col_types = dataset[column].value_counts()
    col_labels = col_types.axes[0].tolist()

    for val in col_labels:
        total = len(dataset[dataset[column] == val].index)
        above = len(dataset[(dataset[column] == val) & (dataset['income'] == '>50K')].index)
        above_list.append(above)
        below_list.append(total - above)

    ind = np.arange(len(col_labels))  # the x locations for the groups

    width = 0.40
    fig, ax = plt.subplots(figsize=(12, 7))
    above_bars = ax.bar(ind, above_list, width, color='#41f474')
    below_bars = ax.bar(ind + width, below_list, width, color='#f44295')

    ax.set_xlabel("Value", fontsize=20)
    ax.set_ylabel('Number of occurrences in dataset', fontsize=20)
    ax.set_title('Feature: {}'.format(column), fontsize=22)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(col_labels,
                       fontsize=7)
    ax.legend((above_bars, below_bars), ('Above 50k', 'Below 50k'), fontsize=17)
    autolabel(ax, above_bars, 10)
    autolabel(ax, below_bars, 10)
    plt.show()

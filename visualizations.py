import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# plot line sustainability coefficient
def plot_sustainability(vec, data_sust, weights_type = '', no = '', title = ''):
    vec = vec * 100
    plt.figure(figsize = (7, 4))
    for j in range(data_sust.shape[0]):
        
        plt.plot(vec, data_sust.iloc[j, :], linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(data_sust.index[j], (x_max + 0.05, data_sust.iloc[j, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel(r'$S$' + ' coefficient [%]', fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xticks(ticks=vec, fontsize = 12)
    plt.title(title)
    plt.grid(linestyle = ':')
    plt.tight_layout()
    plt.savefig('./output_png/' + no + 'sustainability_' + weights_type + '.png')
    plt.savefig('./output/' + no + 'sustainability_' + weights_type + '.pdf')
    pdf = './output/' + no + 'sustainability_' + weights_type + '.pdf'
    return pdf
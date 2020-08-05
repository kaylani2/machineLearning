import matplotlib
import matplotlib.pyplot as plt
import numpy as np


"""
Here we plot the false alarm rate when the attack samples are positive. 
That's why the name pfar.
The results are plotted separately for each dataset, so, if you want
to plot the BoT-IoT resuls you must uncomment where it is said to be done.

"""

# Bezerra2018 results (Value),(Standard Error)


twod_cnn, twod_cnn_std = [(20.96)], [(5.13)]
autoencoder, autoencoder_std = [(4.1)], [(6.74)]
decision_tree, decision_tree_std = [(6.3)], [(1.59)]
rnn, rnn_std = [(60.27)], [(6.01)]
mlp, mlp_std = [(27.2)], [(8.43)]
naive, naive_std = [(53.79)], [(1.99)]
random_forest, random_forest_std = [(37.0)], [(1.15)]



# BoT-IoT results (Value),(Standard Error)

# twod_cnn, twod_cnn_std = [(33.6)], [(12.2)]
# autoencoder, autoencoder_std = [(20.68)], [(6.05)]
# decision_tree, decision_tree_std = [(8.97)], [(2.14)]
# rnn, rnn_std = [(25.78)], [(3.65)]
# mlp, mlp_std = [(12.62)], [(2.49)]
# naive, naive_std = [(25.34)], [(3.61)]
# random_forest, random_forest_std = [(8.3)], [(2.59)]

# ****************
# minimum_value = min(min(naive), min(decision_tree), min(random_forest), min(svm), min(mlp), 
#                     min(twod_cnn), min(autoencoder), min(rnn))
# maximum_value = max(max(naive), max(decision_tree), max(random_forest), max(svm), max(mlp), 
#                     max(twod_cnn), max(autoencoder), max(rnn))

# Chart's bottom (use left on plt.barh if necessary), start from almost the minimum value
# START is quite random, depends on each chart,

# START = round(minimum_value - 5) for starting on the middle. Uncomment lines above for starting at the middle.
# ****************

START = 0

print('Começando em ', START)

# General letters size
SIZE = 40


# Define the lists that will be used to create each bar on the chart
colors_list = ['#B0C4DE', '#F08080', '#90EE90', '#F4A460', '#DDA0DD', '#F0E68C', '#9d67a8']
bars_list = [naive, decision_tree, random_forest, mlp, twod_cnn, autoencoder, rnn]
error_list = [naive_std, decision_tree_std, random_forest_std, mlp_std, twod_cnn_std, 
                autoencoder_std, rnn_std]
label_list = ['Naïve Bayes', 'Árvore de decisão', 'Floresta aleatória', 'MLP', '2DCNN',
                 'Autoencoders', 'LSTM']


# Define the x locations for the groups
ind = np.arange(len(naive))  

# Define the width of the bars
width = 0.1  

# The figsize depends on the user 
fig, ax = plt.subplots(figsize = (11, 6))

for index, group in enumerate(bars_list):
    rects = ax.bar(ind + width * index, np.array(bars_list[index]) - START, width,
                bottom = START,
                yerr = np.array(error_list[index]) * 0.6,
                label = label_list[index],
                color = colors_list[index],
                )


ax.set_xticks(ind + 3.0 * width)
ax.set_xticklabels(['Bezerra2018', 'BoT-IoT'], size = SIZE - 5)

ax.set_yticks(ticks = np.arange(START, 110, 20))
ax.set_yticklabels(['0%', '20%','40%','60%', '80%', '100%'], size = SIZE - 5)

# Legend on the bottom:
ax.legend(bbox_to_anchor=(0., -0.45, 1., .098), loc='center',
           ncol=3, mode="expand", borderaxespad=0., fontsize=SIZE - 19)

# Plot the values inside the bars, the sizes must be adjusted depending on the figure:
for first_index, (my_bar, bar_error) in enumerate(zip(bars_list, error_list)):
    for second_index, (v,std) in enumerate(zip(my_bar, bar_error)):
        ax.text(second_index - 0.01 + first_index/10, START , # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')%').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 18, 
        )


fig.tight_layout()

# # For saving the figure. It was proved to be better to save the figure from the generated one on 
# # plt.show() because it represents the real size.
# plt.savefig("histograma_botiot.eps", # file name
#             dpi = 500,  # dot per inch for resolution increase value for more resolution
#             quality = 100, # "1 <= value <= 100" 100 for best qulity
#             format = 'eps',
#            )

plt.show()

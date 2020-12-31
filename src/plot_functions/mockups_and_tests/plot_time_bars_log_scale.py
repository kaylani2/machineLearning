import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
Here we plot duration of training the models. 
The results are plotted separately for each dataset, so, if you want
to plot the BoT-IoT resuls you must uncomment where it is said to be done.
"""


# #Bezerra2018 results (Value),(Standard Error)

naive, naive_std = ([0.12]), ([0.0])
decision_tree, decision_tree_std = ([1.45]), ([0.04])
random_forest, random_forest_std = ([42.0]), ([2.4])
svm, svm_std = ([1.4]), ([0.1])
mlp, mlp_std = ([36.5]), ([0.2])
twod_cnn, twod_cnn_std = ([187.2]), ([1.67])
autoencoder, autoencoder_std = ([141.06]), ([1.02])
rnn, rnn_std = ([327.2]), ([1.51])


# #BoT-IoT results (Value), (Standard Error)
# naive, naive_std = ([2.22]), ([0.0001])
# decision_tree, decision_tree_std = ([9.21]), ([2.95])
# random_forest, random_forest_std = ([250.0026]), ([7.99])
# svm, svm_std = ([4.76]), ([0.0048])
# mlp, mlp_std = ([517.77]), ([0.0025])
# twod_cnn, twod_cnn_std = ([808.4]), ([1.9])
# autoencoder, autoencoder_std = ([24.05]), ([0.0012])
# rnn, rnn_std = ([113.51]), ([0.0019])

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
width = 0.01

# The figsize depends on the user
# Here we make 2 graphs on the same figure 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5), sharey = True)

# Plotting bars, using logarithm scale
for index, group in enumerate(bars_list):
    rects = ax1.bar(ind + width * index, np.array(bars_list[index]) - START, width,
                bottom = START,
                yerr = np.array(error_list[index]),
                label = label_list[index],
                color = colors_list[index],
                log = True,
                )

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xticks(ind + 3.0 * width)
ax1.set_xticklabels(('Bezerra2018','BoT-IoT'), size = SIZE - 12)

# See output to modify the next line
ax1.set_yticklabels(['  0s', '  0s','  0s','  1s', '  10s', '  100s', '  1000s'], size = SIZE - 10)

# Legend on the bottom:
ax1.legend(bbox_to_anchor=(0., -0.4, 1., .098), loc='center',
           ncol=3, mode="expand", borderaxespad=0., fontsize=SIZE - 23)


# Plot the values inside the bars, the sizes must be adjusted depending on the figure:
for first_index, (my_bar, bar_error) in enumerate(zip(bars_list, error_list)):
    for second_index, (v,std) in enumerate(zip(my_bar, bar_error)):
        ax1.text(second_index + 0.11 + first_index/7.7 - second_index / 2.030 , START,   # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')s').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 18, 
        transform=ax1.transAxes,
        horizontalalignment='center',
        )

# Same thing as the above code, but on the other graph.
# In this case, we are plotting the Bezerra2018 resulted times at left and BoT-IoT at right 

# #BoT-IoT results (Value), (Standard Error)
naive, naive_std = ([2.22]), ([0.0001])
decision_tree, decision_tree_std = ([9.21]), ([2.95])
random_forest, random_forest_std = ([250.0026]), ([7.99])
svm, svm_std = ([4.76]), ([0.0048])
mlp, mlp_std = ([517.77]), ([0.0025])
twod_cnn, twod_cnn_std = ([808.4]), ([1.9])
autoencoder, autoencoder_std = ([24.05]), ([0.0012])
rnn, rnn_std = ([113.51]), ([0.0019])

colors_list = ['#B0C4DE', '#F08080', '#90EE90', '#F4A460', '#DDA0DD', '#F0E68C', '#9d67a8']
bars_list = [naive, decision_tree, random_forest, mlp, twod_cnn, autoencoder, rnn]
error_list = [naive_std, decision_tree_std, random_forest_std, mlp_std, twod_cnn_std, 
                autoencoder_std, rnn_std]
label_list = ['Naïve Bayes', 'Árvore de decisão', 'Floresta aleatória', 'MLP', '2DCNN',
                 'Autoencoders', 'LSTM']

for index, group in enumerate(bars_list):
    rects = ax2.bar(ind + width * index, np.array(bars_list[index]) - START, width,
                bottom = START,
                yerr = np.array(error_list[index]),
                label = label_list[index],
                color = colors_list[index],
                log = True,
                )

ax2.set_xticks(ind + 3.0 * width)
ax2.set_xticklabels(('BoT-IoT', 'Bezerra2018'), size = SIZE - 12)

ax1.set_yticklabels(['  0s', '  0s','  0s','  1s', '  10s', '  100s', '  1000s'], size = SIZE - 10)

for first_index, (my_bar, bar_error) in enumerate(zip(bars_list, error_list)):
    for second_index, (v,std) in enumerate(zip(my_bar, bar_error)):
        ax2.text(second_index + 0.11 + first_index/7.7 - second_index / 2.030 , START,   # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')s').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 18, 
        transform=ax2.transAxes,
        horizontalalignment='center',
        )



# # For saving the figure. It was proved to be better to save the figure from the generated one on 
# # plt.show() because it represents the real size.
# plt.savefig("histograma_botiot.eps", # file name
#             dpi = 500,  # dot per inch for resolution increase value for more resolution
#             quality = 100, # "1 <= value <= 100" 100 for best qulity
#             format = 'eps',
#            )


fig.tight_layout()


plt.show()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
Here we plot the metrics when the attack samples are negative. 
The results are plotted separately for each dataset, so, if you want
to plot the BoT-IoT resuls you must uncomment where it is said to be done.
"""


# # Bezerra2018

twod_cnn, twod_cnn_std = (99.90, 99.89), (0.02, 0.04)
autoencoder, autoencoder_std = (93.87, 68.53), (6.67, 36.60)
decision_tree, decision_tree_std = (99.97, 99.98), (0.01, 0.01)
rnn, rnn_std = (99.7, 99.95), (0.02, 0.01)
mlp, mlp_std = (99.87, 99.90), (0.03, 0.03)
naive, naive_std = (99.74, 98.05), (0.01, 0.02)
random_forest, random_forest_std = (99.82, 99.98), (0.01, 0.01)
svm, svm_std = (94.7, 99.7, 95.03, 97.3, 60.86), (2.06, 0.0, 2.06, 1.1, 2.75)



# # BotIoT results (Value),(Standard Error)

# twod_cnn, twod_cnn_std = (99.99, 99.99), (0.01, 0.01)
# autoencoder, autoencoder_std = (82.99, 99.56), (4.60, 0.30)
# decision_tree, decision_tree_std = (99.99, 100.00), (0.01, 0.00)
# rnn, rnn_std = (99.99, 100.0), (0.01, 0.00)
# mlp, mlp_std = (99.99, 100.0), (0.01, 0.00)
# naive, naive_std = (99.99, 99.97), (0.01, 0.01)
# random_forest, random_forest_std = (99.99, 100.00), (0.01, 0.00)


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
SIZE = 32


# Define the lists that will be used to create each bar on the chart
colors_list = ['#B0C4DE', '#F08080', '#90EE90', '#F4A460', '#DDA0DD', '#F0E68C', '#9d67a8']
bars_list = [naive, decision_tree, random_forest, mlp, twod_cnn, autoencoder, rnn]
error_list = [naive_std, decision_tree_std, random_forest_std, mlp_std, twod_cnn_std, 
                autoencoder_std, rnn_std]
label_list = ['Naïve Bayes', 'Árvore de decisão', 'Floresta aleatória', 'MLP', '2DCNN',
                 'Autoencoders', 'LSTM']


# Define the x locations for the groups
# In this example, we will have 4 groups (PT-BR: 'Acurácia', 'Precisão', 'Sensitivade', 'F1')
#                                         ENG: 'Accuracy', 'Precision', 'Recall', 'F1'
ind = np.arange(len(naive))  

# Define the width of the bars
width = 0.1  

# The figsize depends on the user 
fig, ax1 = plt.subplots(figsize = (15, 7))


for index, group in enumerate(bars_list):
    rects = ax1.bar(ind + width * index, np.array(bars_list[index]) - START, width,
                bottom = START,
                yerr = np.array(error_list[index]),
                label = label_list[index],
                color = colors_list[index],
                )


# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xticks(ind + 3.0 * width)
ax1.set_xticklabels(('Precisão', 'Sensibilidade'), size = SIZE - 0)

ax1.set_yticks(ticks = np.arange(START, 110, 20))
ax1.set_yticklabels(['0%', '20%','40%','60%', '80%', '100%'], size = SIZE - 0)

# Legend on the bottom:
ax1.legend(bbox_to_anchor=(0., -0.40, 1., .098), loc='center',
           ncol=3, mode="expand", borderaxespad=0., fontsize=SIZE - 5)

# Plot the values on the bar, the sizes must be adjusted depending on the figure:
for first_index, (my_bar, bar_error) in enumerate(zip(bars_list, error_list)):
    for second_index, (v,std) in enumerate(zip(my_bar, bar_error)):
        ax1.text(second_index - 0.025 + first_index/10, START , # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')%').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 5, 
        )


fig.tight_layout()




plt.show()

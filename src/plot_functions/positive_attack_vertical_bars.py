import matplotlib
import matplotlib.pyplot as plt
import numpy as np


"""
Here we plot the results when the attack samples are negative. 
The results are plotted separately for each dataset, so, if you want
to plot the BoT-IoT resuls you must uncomment where it is said to be done.
"""


# $ Bezerra2018 results (Value),(Standard Error)


twod_cnn, twod_cnn_std = (99.8, 78.86, 79.03, 78.68), (0.03, 5.94, 5.13, 2.50)
autoencoder, autoencoder_std = (82.21, 79.19, 95.89, 85.69), (17.51, 16.25, 6.74, 10.96)
decision_tree, decision_tree_std = (99.95, 96.08, 93.70, 94.88), (0.01, 0.39, 1.51, 0.90)
rnn, rnn_std = (99.68, 78.60, 43.49, 55.97), (0.01, 2.20, 1.32, 1.18)
mlp, mlp_std = (99.78, 78.61, 72.70, 75.13), (0.01, 4.40, 8.43, 2.49)
naive, naive_std = (97.81, 9.81, 46.20, 16.19), (0.02, 0.43, 1.99, 0.01)
random_forest, random_forest_std = (99.81, 95.30, 62.99, 75.84), (0.01, 0.21, 1.15, 0.81)
svm, svm_std = (94.70, 3.87, 39.13, 7.01), (2.06, 1.06, 2.75, 1.83)



# # Bot-IoT results
# naive, naive_std = (99.9, 63.78, 87.31, 70.00, 74.65) , (0.01, 1.41, 1.91, 1.72, 3.83)
# decision_tree, decision_tree_std = (99.99, 99.99, 100.0, 99.99, 91.02) , (0.01, 0.01, 0.01, 0.01, 2.27)
# random_forest, random_forest_std = (99.99, 99.99, 100.0, 99.99, 91.69) , (0.01, 0.01, 0.01, 0.01, 2.75)
# svm, svm_std = (99.99, 99.99, 100.0, 99.99, 74.65) , (0.01, 0.01, 0.01, 0.01, 3.83)
# mlp, mlp_std = (99.99, 99.99, 100.0, 99.99, 87.37) , (0.01, 0.01, 0.01, 0.01, 2.64)
# twod_cnn, twod_cnn_std = (99.99, 99.99, 99.99, 99.99, 66.39) , (0.01, 0.01, 0.01, 0.01, 12.94)
# autoencoder, autoencoder_std = (89.44, 82.99, 99.56, 90.00, 79.31) , (3.26, 4.61, 0.03, 2.74, 6.42)
# rnn, rnn_std = (99.99, 99.99, 100.0, 99.99, 74.21) , (0.01, 0.00, 0.01, 0.01, 3.87)




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
fig, ax = plt.subplots(figsize = (23, 6))


for index, group in enumerate(bars_list):
    rects = ax.bar(ind + width * index, np.array(bars_list[index]) - START, width,
                bottom = START,
                yerr = np.array(error_list[index]),
                label = label_list[index],
                color = colors_list[index],
                )


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Métricas', size=SIZE/2)
ax.set_xticks(ind + 3.5 * width)
ax.set_xticklabels(('Acurácia', 'Precisão', 'Sensibilidade', 'F1'), size = SIZE - 5)
# ax.xticks(ticks = np.arange(50, 100, 10), labels = '%')

# Y label starting at 50 and ending at 100:
ax.set_yticks(ticks = np.arange(START, 110, 20))
ax.set_yticklabels(['0%', '20%','40%','60%', '80%', '100%'], size = SIZE - 5)

# Legend on the bottom:
ax.legend(bbox_to_anchor=(0., -0.30, 1., .098), loc='center',
           ncol=8, mode="expand", borderaxespad=0., fontsize=SIZE - 11)


# Plot the values inside the bar, the sizes must be adjusted depending on the figure:
for first_index, (my_bar, bar_error) in enumerate(zip(bars_list, error_list)):
    for second_index, (v,std) in enumerate(zip(my_bar, bar_error)):
        ax.text(second_index - 0.03 + first_index/10, START , # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')%').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 11, 
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

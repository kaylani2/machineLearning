import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Bezerra2018 results (Value),(Standard Error)
naive, naive_std =  (99.53, 72.8, 54.98, 58.10) , (0.01, 2.7, 0.01, 0.01)
decision_tree, decision_tree_std =  (99.97, 99.98, 99.98, 99.98) , (0.01, 0.01, 0.01, 0.01)
random_forest, random_forest_std =  (99.53, 99.53, 99.53, 99.76) , (0.01, 0.01, 0.01, 0.01)
svm, svm_std =  (99.67, 99.7, 99.7, 99.83) , (0.02, 0.01, 0.01, 0.01)
mlp, mlp_std =  (99.76, 86.99, 87.10, 86.96) , (0.01, 3.04, 2.35, 1.70)
twod_cnn, twod_cnn_std =  (99.82, 90.01, 91.27, 90.56) , (0.01, 1.88, 2.37, 0.01)
autoencoder, autoencoder_std =  (85.69, 90.25, 90.25, 84.88) , (1.20, 3.56, 3.56, 0.95)
rnn, rnn_std =  (99.6, 99.73, 99.73, 99.84) , (0.01, 0.01, 0.01, 0.01)


# Bot-IoT results
# naive, naive_std = (99.97, 64.72, 87.49, 71.12) , (1.97, 0.01, 0.32, 0.02)
# decision_tree, decision_tree_std = (99.92, 99.26, 99.34, 99.3) , (0.04, 0.62, 0.81, 0.64)
# random_forest, random_forest_std = (99.99, 99.55, 97.14, 98.31) , (4.46, 0.01, 0.01, 0.01)
# svm, svm_std = (99.99, 99.4, 96.3, 97.81) , (0.01, 0.57, 1.06, 0.83)
# mlp, mlp_std = (99.99, 99.7, 87.3, 92.57) , (0.01, 0.03, 2.88, 1.86)
# twod_cnn, twod_cnn_std = (99.99, 93.27, 97.19, 95.13) , (0.01, 1.35, 1.03, 0.02)
# autoencoder, autoencoder_std = (91.13, 85.26, 99.45, 91.81) , (0.01, 0.01, 0.01, 0.01)
# rnn, rnn_std = (99.99, 97.53, 96.07, 96.7) , (0.01, 1.88, 3.23, 1.59)



minimum_value = min(min(naive), min(decision_tree), min(random_forest), min(svm), min(mlp), 
                    min(twod_cnn), min(autoencoder), min(rnn))
maximum_value = max(max(naive), max(decision_tree), max(random_forest), max(svm), max(mlp), 
                    max(twod_cnn), max(autoencoder), max(rnn))

# Chart's bottom (use left on plt.barh if necessary), start from almost the minimum value
# START is quite random, depends on each chart,
START = round(minimum_value - 5) 

# General letters size
SIZE = 32



# Define the lists that will be used to create each bar on the chart
colors_list = ['#B0C4DE', '#F08080', '#90EE90', '#87CEFA', '#F4A460', '#DDA0DD', '#F0E68C', '#9d67a8']

bars_list = [naive, decision_tree, random_forest, svm, mlp, twod_cnn, autoencoder, rnn]

error_list = [naive_std, decision_tree_std, random_forest_std, svm_std, mlp_std, twod_cnn_std, 
                autoencoder_std, rnn_std]

label_list = ['Naïve Bayes', 'Árvore de decisão', 'Floresta aleatória', 'SVM linear', 'MLP', '2DCNN',
                 'Autoencoders', 'LSTM']


# Define the x locations for the groups
# In this example, we will have 4 groups (PT-BR: 'Acurácia', 'Precisão', 'Sensitivade', 'F1')
#                                         ENG: 'Accuracy', 'Precision', 'Recall', 'F1'
ind = np.arange(len(naive))  

# Define the width of the bars
width = 0.1  

# The figsize depends on the user 
fig, ax = plt.subplots(figsize = (23, 6))

# Create a list for charts objects

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
ax.set_xticklabels(('Acurácia', 'Precisão', 'Sensitivade', 'F1'), size = SIZE - 5)
# ax.xticks(ticks = np.arange(50, 100, 10), labels = '%')
ax.set_yticks(np.arange(START, 110, 10))
ax.set_yticklabels(['50%','60%','70%','80%', '90%', '100%'], size = SIZE - 5)

# Legend on the bottom:
ax.legend(bbox_to_anchor=(0., -0.30, 1., .098), loc='center',
           ncol=8, mode="expand", borderaxespad=0., fontsize=SIZE - 14)

# Legend on the right side: 

# ax.legend(loc='center right',
#         bbox_to_anchor=(1.20,0.5),
#         fontsize=SIZE - 13)


# Plot the values on the bar, the sizes must be adjusted depending on the figure:
for first_index, (my_bar, bar_error) in enumerate(zip(bars_list, error_list)):
    for second_index, (v,std) in enumerate(zip(my_bar, bar_error)):
        ax.text(second_index - 0.02 + first_index/10, START , # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')%').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 13, 
        )


fig.tight_layout()


plt.savefig("histograma_bezerra.png", # file name
            dpi = 500,  # dot per inch for resolution increase value for more resolution
            quality = 100, # "1 <= value <= 100" 100 for best qulity
           )

plt.show()
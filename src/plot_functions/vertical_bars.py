import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Bezerra2018 results (Value),(Standard Error)
#naive, naive_std = (32.22, 9.81, 46.2, 16.19), (1.02, 2.23, 1.99, 3.30) #kmeans
naive, naive_std = (27.73, 14.13, 18.94, 16.18), (21.48, 13.22, 13.71, 12.9)
decision_tree, decision_tree_std = (97.08, 96.61, 96.15, 96.38), (0.2, 1.91, 2.23, 0.82)



#decision_tree, decision_tree_std = (99.95, 96.08, 93.7, 94.88), (0.0, 0.39, 1.51, 0.9)

random_forest, random_forest_std = (99.99, 99.27, 91.69, 93.51), (0.01, 0.82, 2.75, 1.40)
mlp, mlp_std = (99.99, 96.01, 87.37, 91.45), (0.01, 1.74, 2.64, 1.10)
twod_cnn, twod_cnn_std = (99.99, 94.26, 66.39, 76.93), (0.01, 6.41, 12.94, 7.25)
#autoencoder, autoencoder_std = (82.21, 79.19, 95.89, 85.69), (17.51, 16.25, 6.74, 10.96)
autoencoder, autoencoder_std = (89.44, 99.44, 79.31, 88.10) , (3.26, 0.41, 6.42, 3.97)
rnn, rnn_std = (99.9, 100.00, 74.21, 85.15), (0.01, 0.00, 3.87, 2.55)



svm, svm_std = (94.7, 3.87, 39.13, 7.01), (2.06, 1.06, 2.75, 1.83)


# # Bot-IoT results
#naive, naive_std = (99.9, 63.78, 87.31, 70.00, 74.65) , (0.01, 1.41, 1.91, 1.72, 3.83)
#decision_tree, decision_tree_std = (99.99, 99.99, 100.0, 99.99, 91.02) , (0.01, 0.01, 0.01, 0.01, 2.27)
#random_forest, random_forest_std = (99.99, 99.99, 100.0, 99.99, 91.69) , (0.01, 0.01, 0.01, 0.01, 2.75)
#svm, svm_std = (99.99, 99.99, 100.0, 99.99, 74.65) , (0.01, 0.01, 0.01, 0.01, 3.83)
#mlp, mlp_std = (99.99, 99.99, 100.0, 99.99, 87.37) , (0.01, 0.01, 0.01, 0.01, 2.64)
#twod_cnn, twod_cnn_std = (99.99, 99.99, 99.99, 99.99, 66.39) , (0.01, 0.01, 0.01, 0.01, 12.94)
#autoencoder, autoencoder_std = (89.44, 82.99, 99.56, 90.00, 79.31) , (3.26, 4.61, 0.03, 2.74, 6.42)
#rnn, rnn_std = (99.99, 99.99, 100.0, 99.99, 74.21) , (0.01, 0.00, 0.01, 0.01, 3.87)



minimum_value = min(min(naive), min(decision_tree), min(random_forest), min(svm), min(mlp),
                    min(twod_cnn), min(autoencoder), min(rnn))
maximum_value = max(max(naive), max(decision_tree), max(random_forest), max(svm), max(mlp),
                    max(twod_cnn), max(autoencoder), max(rnn))

# Chart's bottom (use left on plt.barh if necessary), start from almost the minimum value
# START is quite random, depends on each chart,
# START = round(minimum_value - 5) 
START = 0

print('Começando em ', START)

# General letters size
SIZE = 32


# Define the lists that will be used to create each bar on the chart
# colors_list = ['#B0C4DE', '#F08080', '#90EE90', '#87CEFA', '#F4A460', '#DDA0DD', '#F0E68C', '#9d67a8']
colors_list = ['#B0C4DE', '#F08080', '#90EE90', '#F4A460', '#DDA0DD', '#F0E68C', '#9d67a8']


# bars_list = [naive, decision_tree, random_forest, svm, mlp, twod_cnn, autoencoder, rnn]
bars_list = [naive, decision_tree, random_forest, mlp, twod_cnn, autoencoder, rnn]

# error_list = [naive_std, decision_tree_std, random_forest_std, svm_std, mlp_std, twod_cnn_std, 
#                 autoencoder_std, rnn_std]
error_list = [naive_std, decision_tree_std, random_forest_std, mlp_std, twod_cnn_std, 
                autoencoder_std, rnn_std]


# label_list = ['Naïve Bayes', 'Árvore de decisão', 'Floresta aleatória', 'SVM linear', 'MLP', '2DCNN',
#                  'Autoencoders', 'LSTM']
label_list = ['K-Means', 'SOM', 'Floresta aleatória', 'MLP', '2DCNN',
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
ax.set_xticklabels(('Acurácia', 'Precisão', 'Sensibilidade', 'F1'), size = SIZE - 5)
# ax.xticks(ticks = np.arange(50, 100, 10), labels = '%')

# Y label starting at 50 and ending at 100:
ax.set_yticks(ticks = np.arange(START, 110, 20))
ax.set_yticklabels(['0%', '20%','40%','60%', '80%', '100%'], size = SIZE - 5)

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
        ax.text(second_index - 0.03 + first_index/10, START , # respective position on x and y axis
        (' (' + str(("%.2f" % v)) + ' +/- ' + str(("%.2f" % std)) + ')%').replace('.',','), # String to be plotted
        color='black',
        rotation = 90, # text is plotted horizontally by default 
        fontsize = SIZE - 13, 
        )


fig.tight_layout()


plt.savefig("histograma_bezerra2018_nattack.eps", # file name
            dpi = 500,  # dot per inch for resolution increase value for more resolution
            quality = 100, # "1 <= value <= 100" 100 for best qulity
            format = 'eps',
           )

plt.savefig('output.png')

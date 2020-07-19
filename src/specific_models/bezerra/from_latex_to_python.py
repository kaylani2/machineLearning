# %\begin{table}[ht]
# %\footnotesize
# %\singlespacing
# %    %\setlength\tabcolsep{1.5pt}
# %    \caption{Desempenho dos modelos de aprendizado no conjunto de dados \textit{BoT-IoT}.}
# %    \label{tab:desempenho_bot-iot}
# %    \centering
# %    \begin{tabular}{|l|c|c|c|c|c|c|}
# %        \hline
# %        \thead{\diagbox{Modelo}{Métrica}} & \thead{Acurácia} & \thead{Precisão} & \thead{\textit{Recall}} & \thead{F1} & \thead{Tempo de \\treinamento} \\ \hline
# %        
# %        \thead{Naïve Bayes} & \makecell{$(99,97 \pm$ \\ $1,97)\%$} & \makecell{$(64,72 \pm$ \\ $0,01)\%$} & \makecell{$(87,49 \pm$ \\ $0,32)\%$} & \makecell{$(71,12 \pm$ \\ $0,02)\%$} & \makecell{$(1,78 \pm$ \\ $0,01)$ s} \\ \hline %
# %        \thead{Árvores de \\decisão} & \makecell{$(99,92 \pm$ \\ $0,04)\%$} & \makecell{$(99,26 \pm$ \\ $0,62)\%$} & \makecell{$(99,34 \pm$ \\ $0,81)\%$} & \makecell{$(99,30 \pm$ \\ $0,64)\%$} & \makecell{$(57,14 \pm$ \\ $5,31)$ s} \\ \hline %
# %        \thead{Floresta aleatória} & \makecell{$(99,99 \pm$ \\ $4,46)\%$} & \makecell{$(99,55 \pm$ \\ $0,01)\%$} & \makecell{$(97,14 \pm$ \\ $0,01)\%$} & \makecell{$(98,31 \pm$ \\ $0,01)\%$} & \makecell{$(910,81 \pm$ \\ $79,36)$ s} \\ \hline %
# %        \thead{Máquina de vetores \\de suporte linear} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(99,40\pm$ \\ $0,57)\%$} & \makecell{$(96,30\pm$ \\ $1,06)\%$} & \makecell{$(97,81\pm$ \\ $0,83)\%$} & \makecell{$(38,86\pm$ \\ $1,40)$ s} \\ \hline %
# %        \thead{\textit{Multilayer perceptron}} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(99,70\pm$ \\ $0,03)\%$} & \makecell{$(87,30\pm$ \\ $2,88)\%$} & \makecell{$(92,57\pm$ \\ $1,86)\%$} & \makecell{$(498,30\pm$ \\ $5,31)$ s} \\ \hline %
# %        \thead{Redes neurais convolucionais \\2D por amostra} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(93,27\pm$ \\ $1,35)\%$} & \makecell{$(97,19\pm$ \\ $1,03)\%$} & \makecell{$(95,13\pm$ \\ $0,02)\%$} & \makecell{$(1450,00\pm$ \\ $6,99)$ s} \\ \hline %
# %        \thead{Redes neurais autoassociativas} & \makecell{$(91,13\pm$ \\ $0,01)\%$} & \makecell{$(85,26\pm$ \\ $0,01)\%$} & \makecell{$(99,45\pm$ \\ $0,01)\%$} & \makecell{$(91,81\pm$ \\ $0,01)\%$} & \makecell{$(928,28\pm$ \\ $6,52)$ s} \\ \hline %
# %        \thead{\textit{Long short-term}\\ \textit{memory networks}} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(97,53\pm$ \\ $1,88)\%$} & \makecell{$(96,07\pm$ \\ $3,23)\%$} & \makecell{$(96,70\pm$ \\ $1,59)\%$} & \makecell{$(825,85\pm$ \\ $3,91)$ s} \\ \hline %
# %        
# %    \end{tabular}
# %\end{table}


import re

# str = r'\makecell{$(99,53\pm$ \\ $0,01)\%$} & \makecell{$(72,80\pm$ \\ $2,70)\%$} & \makecell{$(54,98\pm$ \\ $0,01)\%$} & \makecell{$(58,10\pm$ \\ $0,01)\%$} & \makecell{$(0,26\pm$ \\ $0,01)$ s} \\ \hline %'       

# Bezerra
myList = [
        r'\thead{Naïve Bayes} & \makecell{$(99,53\pm$ \\ $0,01)\%$} & \makecell{$(72,80\pm$ \\ $2,70)\%$} & \makecell{$(54,98\pm$ \\ $0,01)\%$} & \makecell{$(58,10\pm$ \\ $0,01)\%$} & \makecell{$(0,26\pm$ \\ $0,01)$ s} \\ \hline %        ',
        r'\thead{Árvore de Decisão} & \makecell{$(99,97\pm$ \\ $0,00)\%$} & \makecell{$(99,98\pm$ \\ $0,00)\%$} & \makecell{$(99,98\pm$ \\ $0,00)\%$} & \makecell{$(99,98\pm$ \\ $0,00)\%$} & \makecell{$(4,46\pm$ \\ $0,38)$ s} \\ \hline%        ',
        r'\thead{Floresta Aleatória} & \makecell{$(99,53\pm$ \\ $0,00)\%$} & \makecell{$(99,53\pm$ \\ $0,00)\%$} & \makecell{$(99,53\pm$ \\ $0,00)\%$} & \makecell{$(99,76\pm$ \\ $0,00)\%$} & \makecell{$(35,32\pm$ \\ $0,27)$ s} \\ \hline%        ',
        r'\thead{Máquina de vetores \\de suporte linear} & \makecell{$(99,67\pm$ \\ $0,02)\%$} & \makecell{$(99,70\pm$ \\ $0,01)\%$} & \makecell{$(99,70\pm$ \\ $0,01)\%$} & \makecell{$(99,83\pm$ \\ $0,01)\%$} &  \makecell{$(187,18\pm$ \\ $217,1)$ s} \\ \hline%        ',
        r'\thead{Multilayer Perceptron} & \makecell{$(99,76\pm$ \\ $0,01)\%$} & \makecell{$(86,99\pm$ \\ $3,04)\%$} & \makecell{$(87,1\pm$ \\ $2,35)\%$} & \makecell{$(86,96\pm$ \\ $1,70)\%$}  & \makecell{$(23,94\pm$ \\ $0,01)$ s} \\ \hline %',
        r'\thead{Redes neurais convolucionais \\ por amostra} & \makecell{$(99,82\pm$ \\ $0,01)\%$} & \makecell{$(90,01\pm$ \\ $1,88)\%$} & \makecell{$(91,27\pm$ \\ $2,37)\%$} & \makecell{$(90,56\pm$ \\ $0,01)\%$} & \makecell{$(832,89\pm$ \\ $3,30)$ s} \\ \hline %',
        r'\thead{Redes neurais autoassociativas} & \makecell{$(85,69\pm$ \\ $1,2)\%$} & \makecell{$(90,25\pm$ \\ $3,56)\%$} & \makecell{$(90,25\pm$ \\ $3,56)\%$} & \makecell{$(84,88\pm$ \\ $0,95)\%$} & \makecell{$(559,50\pm$ \\ $4,96)$ s} \\ \hline%%',
        r'\thead{\textit{Long short-term} \\ \textit{memory networks}}& \makecell{$(99,6\pm$ \\ $0,00)\%$} & \makecell{$(99,73\pm$ \\ $0,00)\%$} & \makecell{$(99,73\pm$ \\ $0,00)\%$} & \makecell{$(99,84\pm$ \\ $0,00)\%$} & \makecell{$(1060,52\pm$ \\ $8,84)$ s} \\ \hline%',
        ]


#BotIoT
myList = [
    r'\thead{Naïve Bayes} & \makecell{$(99,97 \pm$ \\ $1,97)\%$} & \makecell{$(64,72 \pm$ \\ $0,01)\%$} & \makecell{$(87,49 \pm$ \\ $0,32)\%$} & \makecell{$(71,12 \pm$ \\ $0,02)\%$} & \makecell{$(1,78 \pm$ \\ $0,01)$ s} \\ \hline %',
    r'\thead{Árvores de \\decisão} & \makecell{$(99,92 \pm$ \\ $0,04)\%$} & \makecell{$(99,26 \pm$ \\ $0,62)\%$} & \makecell{$(99,34 \pm$ \\ $0,81)\%$} & \makecell{$(99,30 \pm$ \\ $0,64)\%$} & \makecell{$(57,14 \pm$ \\ $5,31)$ s} \\ \hline %',
    r'\thead{Floresta aleatória} & \makecell{$(99,99 \pm$ \\ $4,46)\%$} & \makecell{$(99,55 \pm$ \\ $0,01)\%$} & \makecell{$(97,14 \pm$ \\ $0,01)\%$} & \makecell{$(98,31 \pm$ \\ $0,01)\%$} & \makecell{$(910,81 \pm$ \\ $79,36)$ s} \\ \hline %',
    r'\thead{Máquina de vetores \\de suporte linear} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(99,40\pm$ \\ $0,57)\%$} & \makecell{$(96,30\pm$ \\ $1,06)\%$} & \makecell{$(97,81\pm$ \\ $0,83)\%$} & \makecell{$(38,86\pm$ \\ $1,40)$ s} \\ \hline %',
    r'\thead{\textit{Multilayer perceptron}} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(99,70\pm$ \\ $0,03)\%$} & \makecell{$(87,30\pm$ \\ $2,88)\%$} & \makecell{$(92,57\pm$ \\ $1,86)\%$} & \makecell{$(498,30\pm$ \\ $5,31)$ s} \\ \hline %',
    r'\thead{Redes neurais convolucionais \\ por amostra} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(93,27\pm$ \\ $1,35)\%$} & \makecell{$(97,19\pm$ \\ $1,03)\%$} & \makecell{$(95,13\pm$ \\ $0,02)\%$} & \makecell{$(1450,00\pm$ \\ $6,99)$ s} \\ \hline %',
    r'\thead{Redes neurais autoassociativas} & \makecell{$(91,13\pm$ \\ $0,01)\%$} & \makecell{$(85,26\pm$ \\ $0,01)\%$} & \makecell{$(99,45\pm$ \\ $0,01)\%$} & \makecell{$(91,81\pm$ \\ $0,01)\%$} & \makecell{$(928,28\pm$ \\ $6,52)$ s} \\ \hline %',
    r'\thead{\textit{Long short-term}\\ \textit{memory networks}} & \makecell{$(99,99\pm$ \\ $0,01)\%$} & \makecell{$(97,53\pm$ \\ $1,88)\%$} & \makecell{$(96,07\pm$ \\ $3,23)\%$} & \makecell{$(96,70\pm$ \\ $1,59)\%$} & \makecell{$(825,85\pm$ \\ $3,91)$ s} \\ \hline %',
]       


naive = []
naive_std = []
decision_tree = []
decision_tree_std = []
random_forest = []
random_forest_std = []
svm = []
svm_std = []
mlp = []
mlp_std = []
twod_cnn = []
twod_cnn_std = []
autoencoder = []
autoencoder_std = []
rnn = []
rnn_std = []

plot_list = [
            naive,
            naive_std,
            decision_tree,
            decision_tree_std,
            random_forest,
            random_forest_std,
            svm,
            svm_std,
            mlp,
            mlp_std,
            twod_cnn,
            twod_cnn_std,
            autoencoder,
            autoencoder_std,
            rnn,
            rnn_std
            ]

for first_index, s in enumerate(myList):

    s = s.replace(',','.')
    newstr = ''.join((ch if ch in '0123456789.' else ' ') for ch in s)
    listOfNumbers = [float(i) for i in newstr.split()]
    print(first_index, ':', listOfNumbers)

    for second_index, number in enumerate(listOfNumbers):
       
        if (second_index % 2 == 0): 
            # se x eh par recebe acuracia, precisao etc
            plot_list[2 * first_index].append(number) 
        else: 
            # se x for impar, recebe os desvios 
            plot_list[2 * first_index + 1].append(number)


print('\n\n\n\n\n')

# # Para o primeiro
# print(tuple(naive[0:-1]), ',', tuple(naive_std[0:-1]))
# print(tuple(decision_tree[0:-1]), ',', tuple(decision_tree_std[0:-1]))
# print(tuple(random_forest[0:-1]), ',', tuple(random_forest_std[0:-1]))
# print(tuple(svm[0:-1]), ',', tuple(svm_std[0:-1]))
# print(tuple(mlp[0:-1]), ',', tuple(mlp_std[0:-1]))
# print(tuple(twod_cnn[0:-1]), ',', tuple(twod_cnn_std[0:-1]))
# print(tuple(autoencoder[0:-1]), ',', tuple(autoencoder_std[0:-1]))
# print(tuple(rnn[0:-1]), ',', tuple(rnn_std[0:-1]))


print('(', naive[-1], '),(', naive_std[-1], ')')
print('(', decision_tree[-1], '),(', decision_tree_std[-1], ')')
print('(', random_forest[-1], '),(', random_forest_std[-1], ')')
print('(', svm[-1], '),(', svm_std[-1], ')')
print('(', mlp[-1], '),(', mlp_std[-1], ')')
print('(', twod_cnn[-1], '),(', twod_cnn_std[-1], ')')
print('(', autoencoder[-1], '),(', autoencoder_std[-1], ')')
print('(', rnn[-1], '),(', rnn_std[-1], ')')

# [99.53, 0.01, 72.8, 2.7, 54.98, 0.01, 58.1, 0.01, 0.26, 0.01]
# Accuracy, accstd, precision, prestd, recall, recstd, f1_score, f1std, time, timestd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['SVM', 'MLP', '1D-CNN', 'RNN', 'LSTM']
acc = [89, 92, 94, 91, 99]
acc_std = [5, 9, 2, 11, 2]
recall = [85, 94, 88, 89, 99]

x = np.arange (len (labels))  # the label locations
width = 0.30  # the width of the bars

fig, ax = plt.subplots ()

#plt.errorbar (x - width/2, acc, mfc = 'green', mec = 'red')

rects1 = ax.bar (x - width/2, acc, width, alpha = 0.8, color = 'black',
                 label = 'acc', yerr = acc_std,
                 error_kw = dict (elinewidth = 5, ecolor = 'red'))

rects2 = ax.bar (x + width/2, recall, width, label = 'recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel ('Percentual')
ax.set_title ('Desempenho dos modelos profundos')
ax.set_xticks (x)
ax.set_xticklabels (labels)
ax.legend ()


def autolabel (rects):
  """Attach a text label above each bar in *rects*, displaying its height."""
  for rect in rects:
    height = rect.get_height ()
    ax.annotate ('{}'.format (height),
                xy = (rect.get_x () + rect.get_width () / 2, height),
                xytext = (0, 3),  # 3 points vertical offset
                textcoords = "offset points",
                ha = 'center', va = 'bottom')


autolabel (rects1)
autolabel (rects2)

fig.tight_layout ()
plt.savefig ('bar.png')
plt.show ()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update ({'font.size': 14})
#DDoS              1926624
#DoS               1650260
#Reconnaissance      91082
#Normal                477
#Theft                  79
#labels = ['DDoS', 'DoS', 'Recon.', 'Normal', 'Theft']
#values = [1926624, 1650260, 91082, 477, 79]

#bezerra:


labels = ['Botnets', 'Benigno']
values = [1716408, 7998]
acc_std = [0, 0]

#x = np.arange (len (labels))  # the label locations
x = [0, 1, ]
BAR_WIDTH = 0.50
fig, ax = plt.subplots ()

#plt.errorbar (x - width/2, acc, mfc = 'green', mec = 'red')

rects1 = ax.bar (x, values, BAR_WIDTH, alpha = 0.8, color = 'black',
                 label = '', yerr = acc_std,
                 error_kw = dict (elinewidth = 5, ecolor = 'red'))

#rects2 = ax.bar (x + width/2, recall, width, label = 'recall')



plt.xticks (fontsize = 14)
plt.yticks (fontsize = 14)
plt.ylabel ('Quantidade de amostras', fontsize = 24)
plt.xlabel ('Amostras', fontsize = 24)
#ax.set_ylabel ('Amostras')
ax.set_xticks (x)
ax.set_xticklabels (labels)
ax.set_ylim ([0, 2*1e6])
#ax.set_xlim ([-0.6, 4.6])
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
#autolabel (rects2)

fig.tight_layout ()
plt.savefig ('bar.png')
#plt.show ()

import matplotlib.pyplot as plt

DDOS = 1926624
DOS = 1650260
RECON = 91082
NORMAL = 477
THEFT = 79
TOTAL = DDOS + DOS + RECON + NORMAL + THEFT

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = [ 'DoS', 'Reconnaissance', 'Normal', 'DDos','Theft']
sizes = [ DOS/TOTAL, RECON/TOTAL, NORMAL/TOTAL, DDOS/TOTAL,THEFT/TOTAL]
explode = (0, 0.0, 0, 0.1, 0.2)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



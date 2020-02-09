import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

mpl.use('agg')

# Description: Plots a boxplot of your data points
# Change s1, s2 according ro your data

s1 = [59.0,  1.0,  60.0,  4.0,  1.0,  1.0,  3.0,  61.0,  2.0,  1.0,  1.0,  60.0,  62.0,  2.0,  1.0,  2.0,  59.0,  1.0,  58.0,  55.0,  3.0,  58.0,  60.0,  2.0,  1.0,  58.0,  4.0,  4.0,  1.0,  57.0,  3.0,  59.0,  3.0,  57.0,  6.0,  2.0,  6.0,  1.0,  3.0,  1.0,  4.0,  1.0,  1.0,  1.0,  59.0,  1.0,  57.0,  2.0,  4.0,  57.0,  57.0,  1.0,  2.0,  5.0,  1.0,  56.0,  58.0,  60.0,  1.0,  1.0,  1.0,  1.0,  60.0,  2.0,  61.0,  56.0,  6.0,  1.0,  59.0,  3.0,  57.0,  1.0,  57.0,  3.0,  1.0,  2.0,  57.0,  1.0,  58.0,  59.0,  1.0,  55.0,  5.0,  2.0,  1.0,  1.0,  62.0,  1.0,  4.0,  2.0,  1.0,  56.0,  1.0,  6.0,  2.0,  57.0,  3.0,  6.0,  3.0,  2.0]

s2 = [13.0, 24.0, 28.0, 32.0, 22.0, 28.0, 24.0, 31.0, 11.0, 20.0, 17.0, 22.0, 19.0, 19.0, 17.0, 24.0, 15.0, 20.0, 25.0, 32.0, 21.0, 23.0, 27.0, 21.0, 29.0, 19.0, 34.0, 20.0, 29.0, 22.0, 23.0, 16.0, 7.0, 28.0, 23.0, 29.0, 24.0, 19.0, 15.0, 23.0, 24.0, 29.0, 11.0, 23.0, 33.0, 16.0, 25.0, 12.0, 19.0, 13.0, 25.0, 21.0, 30.0, 22.0, 17.0, 19.0, 22.0, 14.0, 17.0, 31.0, 25.0, 22.0, 26.0, 24.0, 21.0, 5.0, 16.0, 30.0, 16.0, 9.0]


fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
data_to_plot = [s1, s2]

bp = ax.boxplot(data_to_plot, patch_artist=True)

for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)

ax.set_xticklabels(['(1+(4,4))-GA', '(20+20)-GA'])
ax.get_xaxis().tick_bottom()
ax.get_xaxis().tick_top()
ax.get_yaxis().tick_right()


fig.savefig('boxplot.png', bbox_inches='tight')


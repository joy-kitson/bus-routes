import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

path = 'results_12_05_11_25.csv'

util_weight = 1
time_weight = 1

data = pd.read_csv(path)

data['Generation'] = (data['Generation'] + 1) * 5

sns.set_style('whitegrid')
sns.set(font_scale=1.2)
plt.figure(figsize=(30, 30))

for col in ['Max Score', 'Mean Score', 'Min Score', 'Standard Deviation', 'Max Util', 'Min Time', 'Mean Util', 'Mean Time']:
    filename = f'Util{util_weight}_Weight{time_weight}_{col}.png'
    plot_path = os.path.join('..', 'plots', filename)
    g = sns.relplot(x="Generation", y=col, kind="line", data=data)
    g.savefig(plot_path)
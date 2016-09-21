import seaborn as sns
from pandas.tools import plotting
from matplotlib import pyplot as plt

# Load the example tips dataset
iris = sns.load_dataset("iris")

ax = plt.gca()
plt.axis('off')
plotting.table(ax, iris.head(), loc='center')


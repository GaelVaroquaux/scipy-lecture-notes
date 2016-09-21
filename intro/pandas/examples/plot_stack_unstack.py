import seaborn as sns

# Load the example tips dataset
iris = sns.load_dataset("iris")
# Tidy it:
iris = iris.set_index('species').unstack()
iris = iris.reset_index()
iris.columns = ('measure', 'species', 'value')



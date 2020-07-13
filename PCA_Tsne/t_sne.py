import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sn


data = pd.read_csv('train.csv')

l=data['label']

data = data.drop("label",axis=1)

# display or plot a number.
plt.figure(figsize=(7,7))
idx = 8

grid_data = data.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()

print(l[idx])


# Data-preprocessing: Standardizing the data
from sklearn.preprocessing import StandardScaler
standardized_data=StandardScaler().fit_transform(data[0:2500])
print(standardized_data.shape)

l=l[0:2500]


sample_data = standardized_data
Model=TSNE(n_components=2,random_state=0,perplexity=60,learning_rate=250)

tsne_data=Model.fit_transform(sample_data)
# attaching the label for each 2-d data point 
tsne_data = np.vstack((tsne_data.T, l)).T

# creating a new data fram which help us in ploting the result data
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim1", "Dim2", "l"))

#plotting the result tsne
sn.FacetGrid(tsne_df, hue="l", size=6).map(plt.scatter, 'Dim1', 'Dim2').add_legend()
plt.show()




# Machine_Learning

**Dimensanality reduction**

Performed a dimensanality reduction using PCA and t-SNE


Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data.

1. First we want Standardize the given Data.
2. Calculate the covariance matrix X of data points.
3. Calculate eigen vectors and corresponding eigen values.
4. Sort the eigen vectors according to their eigen values in decreasing order.
5. Choose first k eigen vectors and that will be the new k dimensions.
6. Transform the original n dimensional data points into k dimensions.
Eigen Vectors: These vectors gives the in which direction the maximal spread occurs in the data.

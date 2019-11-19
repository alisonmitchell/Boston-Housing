# Boston Housing
The popular Boston Housing dataset contains information collected by the U.S Census Service concerning housing in the area of Boston MA which was originally published in 1978. It is widely used for practice in solving supervised regression problems and is one of the standard datasets included in the scikit-learn library. The challenge is to develop a model that will predict house prices given a set of features that describe houses in Boston.

The notebook demonstrates Exploratory Data Analysis techniques and modelling concepts learned on a face-to-face machine learning course and from various resources including blogs, tutorials, documentation and textbooks.

The dataset is a dictionary so the keys can be printed to view the attributes such as number of samples and features, feature names and characteristics. 

Importing and performing descriptive statistics on the dataset using Pandas revealed two categorical variables, and plotting the distribution of the data using Matplotlib and Seaborn showed a normal distribution with a few outliers. The Pandas corr() function was used to compute the correlation between attributes and Seaborn to visualise the correlation matrix as a heatmap. Univariate and bivariate plots were used to visualise single attributes and two attributes respectively. This included a box plot (univariate) showing outliers whose percentages were calculated using NumPy. 

As part of the preprocessing stage the target variable outliers were removed. After feature selection the dataset was split and stored in features and prices variables respectively. The data was normalised using scikit-learn's MinMaxScaler() function, split into training (80%) and test (20%) subsets and shuffled to remove ordering bias. 
 
During the modelling stage a performance metric (Mean Squared Error) was defined and 11 regression models were evaluated using a 10-fold cross-validation method including linear regression, three regularisation methods, k-nearest neighbors, support vector regressor and decision-tree-based ensemble algorithms. Three models were selected for optimisation using the Grid Search technique and hyperparameters were tuned. Feature importance, mean squared error, and variance score metrics were compared before selecting the Gradient Boosting regressor as the best performing model.

## Data source
[Boston housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) included with the [scikit-learn library](https://scikit-learn.org/stable/datasets/index.html#boston-dataset).  

## Libraries
Numpy, Pandas, SciPy, Scikit-learn, Matplotlib and Seaborn.

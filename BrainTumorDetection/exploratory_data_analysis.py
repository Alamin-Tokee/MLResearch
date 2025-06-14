import numpy as np # calculator library
import pandas as pd # dataframe library
import matplotlib.pyplot as plt # data visualization library
import seaborn as sns # data visualization library based on matplotlib
from collections import Counter
from sklearn.preprocessing import StandardScaler
%matplotlib inline

df = pd.read_csv("/content/drive/MyDrive/Research/Brain/BrainTumor.csv")

df.head()

df.tail()

df.isnull().sum

n = df.isnull().sum()
n

plt.bar(n[0],n[1])

#xticks
plt.xticks(rotation=70)

#x-axis labels
plt.xlabel('Food item')

#y-axis labels
plt.ylabel('Quantity sold')

#plot title
plt.title('Most popular food')
plt.show()

df.isnull().sum().any

class_counts=df.Class.value_counts()
class_counts

df["Class"].value_counts()

x = df.drop(["Image"], axis = 1 )
# y = df["Class"]

x.plot(kind="box",subplots=True,layout=(7,2),figsize=(10,20));

correlations = x.corr()

sns.heatmap(correlations)
plt.show()


def detect_outliers(df, features):
    outlier_indices=[]

    for c in features:
        #1st quartile
        Q1 = np.percentile(df[c],25)
        #3rd quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indices
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers


x.loc[detect_outliers(x,["Class","Mean", "Variance", "Standard Deviation", "Entropy", "Skewness", "Kurtosis", "Contrast", "Energy", "ASM", "Homogeneity", "Dissimilarity", "Correlation", "Coarseness"])]


# drop Outliers
x = x.drop(detect_outliers(x,["Class","Mean", "Variance", "Standard Deviation", "Entropy", "Skewness", "Kurtosis", "Contrast", "Energy", "ASM", "Homogeneity", "Dissimilarity", "Correlation", "Coarseness"]),axis=0).reset_index(drop=True)

correl = x.corr()

sns.heatmap(correl)
plt.show()

x.plot(kind="box",subplots=True,layout=(7,2),figsize=(20,30));

y=x["Class"]
x = x.drop(["Class"], axis=1)
# y = df["Class"]

scalable=['Mean', 'Variance', 'Standard Deviation', 'Entropy',
       'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity',
       'Dissimilarity', 'Correlation', 'Coarseness']


x[scalable]=StandardScaler().fit_transform(x[scalable])
x
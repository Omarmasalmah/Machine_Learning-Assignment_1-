import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your dataset is in a CSV file, change the file path accordingly
file_path = 'cars.csv'

# Load the dataset into a Pandas DataFrame
Dataframe = pd.read_csv(file_path)

warnings.simplefilter(action='ignore', category=FutureWarning)

# 1) Display the number of features and examples
num_examples, num_features = Dataframe.shape
print(f"\nThe dataset has {num_examples} examples and {num_features} features.")

# 2) Calculate the missing values
missing_values = Dataframe.isnull().sum()
print(f"\nMissing values:\n{missing_values}")

# 3) Fill the missing values in each feature using a mean imputation.
# Separate numeric and non-numeric columns
numeric_columns = Dataframe.select_dtypes(include='number').columns
non_numeric_columns = Dataframe.select_dtypes(exclude='number').columns

# Fill missing values for numeric columns using the mean strategy
numeric_imputer = SimpleImputer(strategy='mean')
Dataframe[numeric_columns] = numeric_imputer.fit_transform(Dataframe[numeric_columns])

# Fill missing values for non-numeric columns using the most frequent value (mode)
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
Dataframe[non_numeric_columns] = non_numeric_imputer.fit_transform(Dataframe[non_numeric_columns])

print("Missing values after imputation:")
print(Dataframe.isnull().sum())

# 4) Using Box plot show the mpg for each country that produces cars with better fuel economy.
sns.boxplot(x='origin', y='mpg', data=Dataframe)
plt.title('4. Fuel Economy by Country')
plt.xlabel('Country')
plt.ylabel('Miles Per Gallon (mpg)')
plt.show()

# 5) Which of the following features has a distribution that is most similar to a Gaussian: ‘acceleration’, ‘horsepower’, or ‘mpg’? Answer this part by showing the histogram of each feature.
# Plot the histogram of each feature

sns.histplot(Dataframe['acceleration'])
plt.title('5. Acceleration Distribution')
plt.xlabel('Acceleration')
plt.ylabel('Density')
plt.show()

sns.histplot(Dataframe['horsepower'], color = 'red')
plt.title('5. Horsepower Distribution')
plt.xlabel('Horsepower')
plt.ylabel('Density')
plt.show()

sns.histplot(Dataframe['mpg'], color = 'green')
plt.title('5. MPG Distribution')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.show()

# 7) Plot a scatter plot that shows the ‘horsepower’ on the x-axis and ‘mpg’ on the y-axis. Is there a correlation between them? Positive or negative?
sns.scatterplot(x='horsepower', y='mpg', data=Dataframe)
plt.title('7. Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()










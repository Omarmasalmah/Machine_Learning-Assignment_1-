import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# The path of dataset
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

# 6) Quantitative measure for similarity to Gaussian.

features = ['acceleration', 'horsepower', 'mpg']

# Assuming 'features' is a list of the features you want to analyze
for feature in features:
    # Plot histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(Dataframe[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    
    # Calculate skewness
    skewness = skew(Dataframe[feature])
    
    # Annotate skewness on the plot
    plt.annotate(f'Skewness: {skewness:.2f}', 
                 xy=(0.7, 0.9), 
                 xycoords='axes fraction', 
                 ha='center', 
                 va='center',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white'),
                 fontsize=10)
    
    plt.show()

# 7) Plot a scatter plot that shows the ‘horsepower’ on the x-axis and ‘mpg’ on the y-axis. Is there a correlation between them? Positive or negative?
sns.scatterplot(x='horsepower', y='mpg', data=Dataframe)
plt.title('7. Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.grid()
plt.show()

# 8) Implement the closed form solution of linear regression and use it to learn a linear model to predict the ‘mpg’ from the ‘horsepower’. Plot the learned line on the same scatter plot you got in part 7. (Hint: This is a simple linear regression problem (one feature). Do not forget to add x0=1 for the intercept. For inverting a matrix use np.linalg.inv from NumPy)
X_linear = Dataframe[['horsepower']]
X_linear['x0'] = 1
y_linear = Dataframe['mpg']

w_linear = np.linalg.inv(X_linear.T.dot(X_linear)).dot(X_linear.T).dot(y_linear)

# Scatter plot
#plt.figure(figsize=(8, 6))
sns.scatterplot(x='horsepower', y='mpg', data=Dataframe)
plt.title('8. Linear Regression: Horsepower vs Miles Per Gallon (mpg)')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')

# Plotting the regression line
plt.plot(X_linear['horsepower'], X_linear.dot(w_linear), color='red', linewidth=2)
plt.grid()
plt.show()


# 9) Repeat part 8 but now learn a quadratic function of the form f = w0 + w1x + w2x^2.
X_quadratic = Dataframe[['horsepower']].copy()
X_quadratic['x0'] = 1
X_quadratic['x2'] = X_quadratic['horsepower'] ** 2
y_quadratic = Dataframe['mpg'].copy()

w_quadratic = np.linalg.inv(X_quadratic.T.dot(X_quadratic)).dot(X_quadratic.T).dot(y_quadratic)
# Scatter plot
#plt.figure(figsize=(8, 6))
sns.scatterplot(x='horsepower', y='mpg', data=Dataframe)
plt.title('9. Quadratic Regression: Horsepower vs Miles Per Gallon (mpg)')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')
# Plotting the regression line
plt.plot(X_quadratic['horsepower'], X_quadratic.dot(w_quadratic), color='red', linewidth=2)
plt.grid()
plt.show()


# 10) Gradient descent

def gradient_descent(X, y, w, alpha, iterations): # 
    for i in range(iterations): 
        w = w - alpha * (2 * X.T.dot(X.dot(w) - y) / len(y) ) # Update w using the gradient and the learning rate 
    return w

# Additional preprocessing steps to check for NaN or infinite values and normalize data
for column in Dataframe.columns:
    column_data = Dataframe[column].values  # Convert column to NumPy array
    if column != 'origin' and (np.any(np.isnan(column_data.astype(float))) or np.any(np.isinf(column_data.astype(float)))):
        print(f'Warning: NaN or infinite values detected in column {column}.')
    elif column != 'origin':
        # Normalize the data
        Dataframe[column] = (Dataframe[column] - np.min(column_data)) / (np.max(column_data) - np.min(column_data))

# Prepare data for linear regression
X_linear = Dataframe[['horsepower']].copy()
X_linear['x0'] = 1
y_linear = Dataframe['mpg'].copy()

# Initial parameters
weight_initial = np.zeros(X_linear.shape[1])

learning_rate = 0.1
iterations = 1500

# Run gradient descent
weight_final = gradient_descent(X_linear, y_linear, weight_initial, learning_rate, iterations)

# Scatter plot
sns.scatterplot(x='horsepower', y='mpg', data=Dataframe)
plt.title('10. Linear Regression: Horsepower vs Miles Per Gallon (mpg) - Gradient Descent')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')

# Plotting the regression line
plt.plot(X_linear['horsepower'], X_linear.dot(weight_final), color='red', linewidth=2)
plt.grid()
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
lifesat = pd.read_csv("data/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

if __name__ == '__main__':
    # Make a prediction for Cyprus
    X_new = [[37_655.2]]  # Cyprus' GPD per capita in 2020
    print(model.predict(X_new))  # output: [[6.30165767]]

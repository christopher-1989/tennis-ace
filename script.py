#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
print(df.head())
columns = df.columns


# perform exploratory analysis here:






















## perform single feature linear regressions here:

x = df[['Aces']]
y = df[['wins']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()




















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:

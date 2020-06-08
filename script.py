#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
#print(df.head())
columns = df.columns

# perform exploratory analysis here:
print(columns)
#print(df[['Wins']].describe())
#print(df[['Losses']].describe())

#print(df[['Aces']].describe())


# perform single feature linear regressions here:

x = df[['Aces']]
y = df[['Wins']]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("One feature linear regression Train score:")
print(lm.score(x_train, y_train))

print("One feature linear regression Test score:")
print(lm.score(x_test, y_test))
plt.scatter(y_test, y_predict)
plt.xlabel("Actual win:")
plt.ylabel("Predicted win:")
plt.title("Actual win vs Predicted win")
print(model.coef_)
plt.show()


## perform two feature linear regressions here:


x = df[['Aces', 'DoubleFaults']]
y = df[['Wins']]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Two feature linear regression Train score:")
print(lm.score(x_train, y_train))

print("Two feature linear regression Test score:")
print(lm.score(x_test, y_test))
plt.scatter(y_test, y_predict)
plt.xlabel("Actual win:")
plt.ylabel("Predicted win:")
plt.title("Actual win vs Predicted win")
print(model.coef_)
plt.show()


## perform multiple feature linear regressions here:
x = df[['Year', 'FirstServe', 'FirstServePointsWon',
       'FirstServeReturnPointsWon', 'SecondServePointsWon',
       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted',
       'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
       'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
       'TotalPointsWon', 'TotalServicePointsWon', 'Losses', 'Winnings',
       'Ranking']]
y = df[['Wins']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Multiple linear regression train score:")
print(lm.score(x_train, y_train))

print("Multiple linear regression test score:")
print(lm.score(x_test, y_test))
plt.scatter(y_test, y_predict)
plt.xlabel("Actual win:")
plt.ylabel("Predicted win:")
plt.title("Actual win vs Predicted win")
print(model.coef_)
plt.show()

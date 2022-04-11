import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randint(0,100,size=(100, 2)), columns=list('AB'))
df.plot.scatter(x='A', y='B')
df['C'] = 2*df['A'] + 3
df.plot.scatter(x='A', y='C')

noise = np.random.normal(0, 1, 100)
df['D'] = pd.Series(noise)
df['E'] = df['C'] + df['D']
df.plot.scatter(x='A', y='E')

x = df['A'].to_numpy().reshape(len(df), 1)
X = np.append(x, np.ones((len(df), 1)), axis=1)
y = df['C'].to_numpy().reshape(len(df), 1)

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
df['y_pred'] = theta[0]*x + theta[1]
df.plot.scatter(x='A', y='E')
plt.plot(x, df['y_pred'])
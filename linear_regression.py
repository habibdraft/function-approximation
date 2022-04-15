#!/usr/bin/env python
# coding: utf-8

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randint(0,100,size=(100, 2)), columns=list('AB'))
df.plot.scatter(x='A', y='B')
df['C'] = 2*df['A'] + 3
df.plot.scatter(x='A', y='C')

noise = np.random.normal(0, 5, 100)
df['D'] = pd.Series(noise)
df['E'] = df['C'] + df['D']
df.plot.scatter(x='A', y='E')

x = df['A'].to_numpy().reshape(len(df), 1)
X = np.append(x, np.ones((len(df), 1)), axis=1)
y = df['E'].to_numpy().reshape(len(df), 1)

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = theta[0]*x + theta[1]

df.plot.scatter(x='A', y='E')
plt.plot(x, y_pred)

MSE = np.square(np.subtract(y, y_pred)).mean()
SSE = np.sum(np.square(np.subtract(y, y_pred)))
SSR = np.sum(np.square(np.subtract(y.mean(), y_pred)))
SST = SSR + SSE
R_squared = SSR / SST






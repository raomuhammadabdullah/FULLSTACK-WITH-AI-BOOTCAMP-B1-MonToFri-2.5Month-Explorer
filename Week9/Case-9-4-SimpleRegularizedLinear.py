import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.DataFrame.from_dict({
    'x': [1,2,3], 
    'y': [1,2,4]})

x = df[['x']]
y = df.y

rdg = Ridge(alpha = 0.1)
rdg.fit(x, y)

y_pred = rdg.predict(x)
fig = plt.figure(figsize=(15,10))
plt.plot(df['x'],df['y'],'o',color="green", markersize=20)
plt.plot(df['x'],y_pred, linewidth=10,label = "y = {:.2f}x + {:.2f}".format(float(rdg.coef_),float(rdg.intercept_)))
plt.xlabel("x", fontsize=20)
plt.ylabel("y",  fontsize=20)
plt.legend(loc="upper left", fontsize = 20)
plt.show()

wait = input( "Wait here...")
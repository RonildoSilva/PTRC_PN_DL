import numpy as np
import scipy.stats as st
import pandas as pd

def ci(data):
  #create 95% confidence interval for population mean weight
  return st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
  
df = pd.read_csv('Baseline AL_CG_DummyRegressor.csv')
y_test = []

for t in df['y_test'].values:
  str_v = t.replace('[','')
  str_v = str_v.replace(']','')  
  y_test.append(float(str_v))

y_pred = np.squeeze(df['y_pred']).values


int_conf = ci(abs(y_test - y_pred))

print(int_conf)

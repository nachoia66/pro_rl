import pandas as pd
import numpy as np

y_test=np.array([2,3,5,7,10])
y_pred=y_test*3000
datos={'Test':y_test,'Predicho':y_pred}
df=pd.DataFrame(data=datos)
df.head(6)
from mlc.datasets.dataset_factory import get_dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

ds = get_dataset('lcld_v2_time')
splits = ds.get_splits()

metadata = ds.get_metadata(only_x=True)
feature_types = metadata['type'].to_list()
print(f'feature_types {np.unique(np.array(feature_types), return_counts=True)}')

x, y = ds.get_x_y()
#x = x.to_numpy()
print(f'x shape {x.shape}')
#x_test, y_test = x[splits['test']], y[splits['test']]
categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']
for cat in categorical:
    print(f'unique values for {cat} : {pd.unique(x[cat])}')
encoded_df = pd.get_dummies(x, columns=categorical)
encoded_df = encoded_df.to_numpy()
x_train, y_train = encoded_df[splits['train']], encoded_df[splits['train']]
print(encoded_df.shape)
#for c in categorical :
    #x[c] = pd.get_dummies(x[c])

print(f'new shape {x.shape}')

scaler = MinMaxScaler()
scaler.fit(x_train)
joblib.dump(scaler, './ressources/custom_lcld_scaler.joblib')

#print(x_test.shape, y_test.shape)
#print(np.unique(y_test, return_counts=True))

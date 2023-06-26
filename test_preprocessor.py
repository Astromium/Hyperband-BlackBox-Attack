import joblib
from mlc.datasets.dataset_factory import get_dataset
import numpy as np

ds = get_dataset('lcld_v2_iid')
x, y = ds.get_x_y()
adversarials = np.load('./adversarials_lcld.npy')

categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']
cat_indices = [x.columns.get_loc(col) for col in categorical]


x = x.to_numpy()

preprocessor = joblib.load('./ressources/lcld_preprocessor.joblib')

x1 = x[0]

print(x1.shape)

x1_scaled = preprocessor.transform(x1.reshape(1, -1))

print(x1_scaled.shape)


categorical_indices = preprocessor.transformers_[1][2]
numerical_indices = preprocessor.transformers_[0][2]

print(f'cat indices {cat_indices}')
print(f'cat {categorical_indices}')
print(f'num {numerical_indices}')


one_hot_columns = x1_scaled[:, categorical_indices]
print(f'one hot columns {one_hot_columns}')
print(f'{preprocessor.transformers_[1][1].categories_[0]}')
original_categorical = preprocessor.transformers_[1][1]['onehot'].categories_[0][one_hot_columns.argmax(axis=1)]

original_numerical = x1_scaled[:, numerical_indices]

x1_rescaled = np.column_stack((original_numerical, original_categorical))

print(f'Re-transformed shape {x1_rescaled.shape}')

print(f'cat cols original {x1.reshape(1, -1)[:, categorical_indices]}')
print(f'cat cols re-transformed {x1_rescaled[:, categorical_indices]}')
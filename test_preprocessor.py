import joblib
from mlc.datasets.dataset_factory import get_dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from utils.inverse_transform import inverse_transform

ds = get_dataset('lcld_v2_iid')
x, y = ds.get_x_y()
categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']

#x[categorical] = x[categorical].astype(str)

numerical = [col for col in x.columns if col not in categorical]
num_indices = [x.columns.get_loc(col) for col in numerical]
col_order = list(numerical) + list(categorical)
x = x[col_order]
cat_indices = [x.columns.get_loc(col) for col in categorical]
print(f'cat indices {cat_indices}')

num_transformer = MinMaxScaler()
cat_transformer = OneHotEncoder(sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_indices),
        ('cat', cat_transformer, cat_indices)
    ]
)

x = x.to_numpy()
x1 = x[0, :]
print(f'original cat features of x1 : {x1[cat_indices]}')
print(f'original num features of x1 : {x1[num_indices]}')
preprocessor.fit(x)
x = preprocessor.transform(x)

print(x.shape)

ohe = preprocessor.transformers_[1][1]
scaler = preprocessor.transformers_[0][1]
x1 = x[0, :]
print(f'categories {preprocessor.transformers_[1][1].categories_}')
print(f'cat features of transformed x1 : {x1[preprocessor.transformers_[1][2][0]:]}')

categories = preprocessor.transformers_[1][1].categories_
c = preprocessor.transformers_[1][2][0]
print(f'c {c}')
maxs = []
for i in range(len(categories)):
    arr = x1[c:c+len(categories[i])]
    maxs.append(np.argmax(arr))
    c += len(categories[i])

print(f'maxs {maxs}')

num_rescaled = scaler.inverse_transform(x1[:preprocessor.transformers_[1][2][0]].reshape(1, -1))
print(f'num rescaled {num_rescaled}')

rescaled_x1 = np.concatenate((num_rescaled[0], np.array(maxs)))

print(f'rescaled_x1 {rescaled_x1}')

x1_inv = inverse_transform(preprocessor=preprocessor, x=x1)
print(f'x1_inv {x1_inv}')    
'''
adversarials = np.load('./adversarials_lcld.npy')

categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']
cat_indices = [x.columns.get_loc(col) for col in categorical]


x = x.to_numpy()

preprocessor = joblib.load('./ressources/lcld_preprocessor.joblib')

x1 = x[0]

print(x1.shape)

x1_scaled = preprocessor.transform(x1.reshape(1, -1))

print(f'x1_scaled {x1_scaled}')

categorical_indices = preprocessor.transformers_[1][2]
numerical_indices = preprocessor.transformers_[0][2]

categorical_indices = preprocessor.transformers_[1][2]
numerical_indices = preprocessor.transformers_[0][2]

print(f'cat indices {cat_indices}')
print(f'cat {categorical_indices}')
print(f'num {numerical_indices}')

categories = preprocessor.transformers_[1][1].categories_
print(f'categories {categories[0]}')
cols_after_encoding = []
for i in range(len(categories)):
    start = categories
    arr = list(range(categorical_indices[i], categorical_indices[i]+len(categories[i])))
    cols_after_encoding.extend(arr)
print(f'cols_after_encoding {len(cols_after_encoding)}')
ohe = preprocessor.transformers_[1][1]
x11 = ohe.transform(x1.reshape(1, -1)[:, cat_indices])
x11_rescaled = ohe.inverse_transform(x11)
print(f'x11 {x11}')
print(f'x11_rescaled {x11_rescaled}')
print(f'ohe {ohe}')
x1_rescaled = ohe.inverse_transform(x1_scaled[:, cols_after_encoding])
print(f'x1_rescaled {x1_rescaled}')
print(f'x1 {x1[cat_indices]}')

one_hot_columns = x1_scaled[:, categorical_indices[0]:len(categories[0])+categorical_indices[0]]
print(f'one hot columns {one_hot_columns}')
print(f'categories_ : {preprocessor.transformers_[1][1].categories_}')
#original_categorical = preprocessor.transformers_[1][1]['onehot'].categories_[0][one_hot_columns.argmax(axis=1)]

original_numerical = x1_scaled[:, numerical_indices]

x1_rescaled = np.column_stack((original_numerical, original_categorical))

print(f'Re-transformed shape {x1_rescaled.shape}')

print(f'cat cols original {x1.reshape(1, -1)[:, categorical_indices]}')
print(f'cat cols re-transformed {x1_rescaled[:, categorical_indices]}')

'''
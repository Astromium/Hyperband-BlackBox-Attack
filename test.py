from mlc.datasets.dataset_factory import get_dataset
from sklearn.preprocessing import MinMaxScaler
import joblib

ds = get_dataset("ctu_13_neris")
X, _ = ds.get_x_y()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)

joblib.dump(scaler, './ressources/custom_botnet_scaler.joblib')

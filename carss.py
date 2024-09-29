import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Đọc dữ liệu từ file CSV
cars = pd.read_csv('CarPrice_Assignment.csv')

# Tách đặc trưng (features) và nhãn (target)
X = cars.drop(['price', 'CarName', 'car_ID'], axis=1)
y = cars['price']

# One-hot encoding cho các cột dạng object
X = pd.get_dummies(X, drop_first=True)

# Chia dữ liệu thành train và test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện các mô hình
lin_reg = LinearRegression().fit(X_train_scaled, y_train)
ridge_reg = Ridge(alpha=5.0).fit(X_train_scaled, y_train)

# Huấn luyện MLPRegressor với các tham số điều chỉnh

nn_reg = MLPRegressor(hidden_layer_sizes=(150, 75), max_iter=100000, learning_rate_init=0.01, 
                      random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10).fit(X_train_scaled, y_train)

# Bạn có thể giữ các phần còn lại của mã không thay đổi


# Stacking model
stacking_reg = StackingRegressor(
    estimators=[('lr', lin_reg), ('ridge', ridge_reg), ('nn', nn_reg)],
    final_estimator=DecisionTreeRegressor()
).fit(X_train_scaled, y_train)

# Lưu các mô hình và scaler
with open('models.pkl', 'wb') as f:
    pickle.dump((scaler, lin_reg, ridge_reg, nn_reg, stacking_reg), f)

# Định nghĩa hàm dự đoán
def predict_price(data):
    input_data = pd.DataFrame([data])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)

    predictions = {
        'linear_regression': lin_reg.predict(input_data_scaled)[0],
        'ridge_regression': ridge_reg.predict(input_data_scaled)[0],
        'neural_network': nn_reg.predict(input_data_scaled)[0],
        'stacking_model': stacking_reg.predict(input_data_scaled)[0],
    }
    
    return predictions

# Ví dụ sử dụng với dữ liệu mới
new_data = {
    'wheelbase': 98.4,
    'carlength': 168.8,
    'carwidth': 64.1,
    'curbweight': 2548,
    'enginesize': 130,
    'horsepower': 111,
    'peakrpm': 5000,
    'citympg': 21,
    'highwaympg': 27,
}

# Dự đoán giá
predicted_prices = predict_price(new_data)
print(predicted_prices)

# Dự đoán trên tập huấn luyện và tập kiểm tra
train_predictions = nn_reg.predict(X_train_scaled)
test_predictions = nn_reg.predict(X_test_scaled)

# Tính RMSE, MAE và R²
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

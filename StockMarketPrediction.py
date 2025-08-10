import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import talib as ta

# 1. Tải và chuẩn bị dữ liệu
sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="max")
sp500_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)
sp500_data = sp500_data.loc["1990-01-01":].copy()

# 2. Thêm các chỉ báo kỹ thuật
def add_technical_indicators(df):
    # Chỉ báo xu hướng
    df['MA10'] = ta.MA(df['Close'], timeperiod=10)
    df['MA50'] = ta.MA(df['Close'], timeperiod=50)
    df['MA200'] = ta.MA(df['Close'], timeperiod=200)
    
    # Chỉ báo động lượng 
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'])
    
    # Chỉ báo biến động
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Chỉ báo khối lượng
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])
    
    df.fillna(method='bfill', inplace=True)
    return df

sp500_data = add_technical_indicators(sp500_data)

# 3. Chia dữ liệu theo thời gian (không xáo trộn)
split_date = '2020-01-01'
train = sp500_data.loc[:split_date].copy()
test = sp500_data.loc[split_date:].copy()

# 4. Chuẩn bị features và target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'MA10', 'MA50', 'MA200', 'RSI', 'MACD', 
            'MACD_Signal', 'ATR', 'OBV']

# 5. Tiền xử lý dữ liệu riêng biệt
scaler = RobustScaler()  # Ít nhạy cảm với ngoại lệ hơn StandardScaler
train_scaled = scaler.fit_transform(train[features])
test_scaled = scaler.transform(test[features])

# 6. Tạo sequences với cửa sổ thời gian
def create_sequences(data, targets, window_size=90):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, train['Target'], window_size=90)
X_test, y_test = create_sequences(test_scaled, test['Target'], window_size=90)

# 7. Xây dựng mô hình LSTM cải tiến
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, 
                      kernel_regularizer=l2(0.001),
                      input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.4),
    Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 8. Tối ưu hóa và callback
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['accuracy', 
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# 9. Huấn luyện với TimeSeriesSplit để validate
val_size = int(0.2 * len(X_train))  # 20% cuối làm validation

X_train_final = X_train[:-val_size]
y_train_final = y_train[:-val_size]
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]

history = model.fit(
    X_train_final, y_train_final,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 10. Đánh giá mô hình
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# 11. Dự đoán
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)
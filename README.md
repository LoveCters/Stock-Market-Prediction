

# S&P 500 Price Direction Prediction using LSTM

##  Overview
This project builds a **Bidirectional LSTM** model to predict the **next-day price direction** of the **S&P 500 index** (^GSPC).  
The model uses historical market data along with various **technical indicators** to forecast whether the market will go **up** or **down** the next day.

---

##  Dataset
- Source: [`yfinance`](https://pypi.org/project/yfinance/) (Yahoo Finance API)
- Symbol: `^GSPC` (S&P 500 Index)
- Date range: From **1990-01-01** to present
- Data includes:
  - Open, High, Low, Close, Volume
  - Technical Indicators (MA, RSI, MACD, ATR, OBV)

---

## Features
| Feature | Description |
|---------|-------------|
| `MA10`, `MA50`, `MA200` | Moving Averages |
| `RSI` | Relative Strength Index |
| `MACD`, `MACD_Signal` | MACD and signal line |
| `ATR` | Average True Range |
| `OBV` | On-Balance Volume |

---

## Model Architecture
The model is a **Bidirectional LSTM** network with dropout and L2 regularization to reduce overfitting.

**Layers:**
1. `Bidirectional LSTM (128 units)` — with L2 regularization, return sequences
2. `Dropout (0.4)`
3. `Bidirectional LSTM (64 units)` — with L2 regularization
4. `Dropout (0.3)`
5. `Dense (32 units, ReLU)` — with L2 regularization
6. `Dropout (0.2)`
7. `Dense (1 unit, Sigmoid)` — binary classification output

**Loss & Metrics:**
- Loss: `binary_crossentropy`
- Metrics: `accuracy`, `precision`, `recall`


## Training Setup
- **Train/Validation/Test Split**:
  - Train: Data up to `2020-01-01`
  - Test: Data from `2020-01-01` onwards
  - 20% of the train set is used for validation
- **Scaling**: `RobustScaler` (less sensitive to outliers)
- **Sequence Length**: 90 days
- **Optimizer**: Adam (`learning_rate=0.0005`)
- **Callbacks**:
  - EarlyStopping (patience=15)
  - ReduceLROnPlateau (patience=5, factor=0.2)


## Evaluation Metrics
The model reports:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Example output:

Test Accuracy: 0.5379
Test Precision: 0.5542
Test Recall: 0.7580
Test F1-Score: 0.6403



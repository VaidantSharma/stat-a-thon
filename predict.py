import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Bidirectional, TimeDistributed, Layer
from tensorflow.keras.layers import GlobalAveragePooling1D, Add, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
#from tensorflow.keras.optimizers.schedules import CosineDecay  # Optional: use cosine decay scheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

######################################
# 1. Data Loading & Feature Engineering
######################################

def load_and_preprocess(file_path):
    # Load data with header (assumes first row as header)
    df = pd.read_csv(file_path, header=0)
    # Convert 'Date' to datetime; adjust format as needed
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')
    
    # Clean numeric columns (remove commas) for Price, Open, High, Low
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # Handle volume column with M and K suffixes
    def convert_volume(vol_str):
        vol_str = str(vol_str).replace(',', '')
        if 'M' in vol_str:
            return float(vol_str.replace('M', ''))
        elif 'K' in vol_str:
            return float(vol_str.replace('K', '')) / 1000
        else:
            return float(vol_str)
    
    df['Vol.'] = df['Vol.'].apply(convert_volume)
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    
    return df

def create_features(df):
    # Create return and log-return features(How much the price has changed compared to previous days.)
    df['Return'] = df['Price'].pct_change()
    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Moving averages and exponential moving averages(Simple and exponential moving averages (e.g., MA5 = average over 5 days).)
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Price'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Price'].ewm(span=window, adjust=False).mean()
    
    # Volatility features and Bollinger Bands (using 10 and 20 days windows)(Measures volatility)
    for window in [10, 20]:
        df[f'STD_{window}'] = df['Price'].rolling(window=window).std()
        df[f'Upper_BB_{window}'] = df[f'MA_{window}'] + 2 * df[f'STD_{window}']
        df[f'Lower_BB_{window}'] = df[f'MA_{window}'] - 2 * df[f'STD_{window}']
        df[f'BB_Width_{window}'] = (df[f'Upper_BB_{window}'] - df[f'Lower_BB_{window}']) / df[f'MA_{window}']
    
    # Price momentum indicators(Rate of Change)
    df['ROC_5'] = df['Price'].pct_change(periods=5) * 100
    df['ROC_10'] = df['Price'].pct_change(periods=10) * 100
    
    # RSI Calculation( (Relative Strength Index): Measures the speed and change of price movements.)
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD(A trend-following momentum indicator. Relation between two moving average)
    df['MACD'] = df['Price'].ewm(span=12, adjust=False).mean() - df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Candle patterns – 'Price'
    df['Body_Size'] = abs(df['Price'] - df['Open']) / ((df['Price'] + df['Open']) / 2) * 100
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Price']].max(axis=1)) / ((df['Price'] + df['Open']) / 2) * 100
    df['Lower_Shadow'] = (df[['Open', 'Price']].min(axis=1) - df['Low']) / ((df['Price'] + df['Open']) / 2) * 100
    
    # Normalized price relative to MA20(how far price is from the 20-day moving average.)
    df['Price_to_MA20'] = df['Price'] / df['MA_20'] - 1
    
    df = df.dropna()
    return df

def create_sequences(data, target, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(target[i+time_steps])
    return np.array(X), np.array(y)

######################################
# 2. Custom KAN Layer (Simplified Version)
######################################

class KANLayer(Layer):
    """
    A simplified KAN-like layer that learns a univariate cubic polynomial per input feature.
    For each feature x, it computes: f(x) = a*x^3 + b*x^2 + c*x + d.
    Then, for each output dimension, sums over all input features.
    """
    def __init__(self, output_dim, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # input_shape: (batch_size, n_features)
        self.n_features = input_shape[-1]
        # Create trainable coefficients for a cubic polynomial per feature and per output dimension.
        # Shape: (n_features, output_dim, 4) where the last dimension corresponds to [a, b, c, d]
        self.coeffs = self.add_weight(shape=(self.n_features, self.output_dim, 4),
                                      initializer='glorot_uniform',
                                      trainable=True,
                                      name='coeffs')
        super(KANLayer, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, inputs):
        x_exp = tf.expand_dims(inputs, axis=-1)  # (batch_size, n_features, 1)
        # Compute polynomial terms: x^3, x^2, x, and 1.
        x1 = x_exp
        x2 = tf.math.pow(x_exp, 2)
        x3 = tf.math.pow(x_exp, 3)
        ones = tf.ones_like(x_exp)
        # Concatenate to shape (batch_size, n_features, 4)
        poly_terms = tf.concat([x3, x2, x1, ones], axis=-1)
        # Expand dims to align with coeffs: poly_terms -> (batch, n_features, 1, 4)
        poly_terms_exp = tf.expand_dims(poly_terms, axis=2)
        # Expand coeffs: (1, n_features, output_dim, 4)
        coeffs_exp = tf.expand_dims(self.coeffs, axis=0)
        # Multiply and sum over polynomial terms: resulting shape (batch, n_features, output_dim)
        transformed = tf.reduce_sum(poly_terms_exp * coeffs_exp, axis=-1)
        # Sum over input features: (batch, output_dim)
        output = tf.reduce_sum(transformed, axis=1)
        return output

######################################
# 3. Build the Hybrid Model (KAN + BiLSTM + MultiHeadAttention)
######################################

def build_hybrid_model(input_shape):
    # input_shape: (time_steps, n_features)
    inputs = Input(shape=input_shape)
    # Apply TimeDistributed KAN layer: apply the KAN on each time step independently.
    kan_out = TimeDistributed(KANLayer(output_dim=32))(inputs)
    kan_out = BatchNormalization()(kan_out)
    
    # Bidirectional LSTM layers for temporal modeling.
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(kan_out)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    lstm_out2 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(lstm_out)
    lstm_out2 = Dropout(0.3)(lstm_out2)
    lstm_out2 = BatchNormalization()(lstm_out2)
    
    # MultiHead Attention block with a residual connection.
    mha = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out2, lstm_out2)
    mha_out = Add()([lstm_out2, mha])
    mha_out = BatchNormalization()(mha_out)
    
    # Global pooling to summarize the sequence information.
    gap = GlobalAveragePooling1D()(mha_out)
    
    # Dense layers for final prediction.
    dense = Dense(64, activation='relu')(gap)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(32, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    outputs = Dense(1)(dense)
    
    # Optionally, try a cosine decay learning rate scheduler.
    # lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000, alpha=0.0001)
    # optimizer = Adam(learning_rate=lr_schedule)
    optimizer = Adam(learning_rate=0.001)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='huber')
    return model

######################################
# 4. Plotting Training History
######################################

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

######################################
# 5. Main Function: Data Preparation, Model Training & Prediction
######################################

def main(file_path, strike_price, expiration_date):
    print("Loading and preprocessing data...")
    df = load_and_preprocess(file_path)
    print(f"Loaded dataframe shape: {df.shape}")
    
    print("Creating features...")
    df = create_features(df)
    print(f"Feature dataframe shape: {df.shape}")
    
    if len(df) == 0:
        print("No data available after preprocessing.")
        return
    
    # Exclude 'Date' column for features.
    feature_cols = [col for col in df.columns if col != 'Date']
    
    # Fill any remaining NaNs with column means.
    if df[feature_cols].isna().any().any():
        df = df.fillna(df.mean())
    
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Save price scaler if needed.
    price_scaler = StandardScaler()
    price_scaler.fit_transform(df[['Price']])
    
    print("Creating sequences...")
    X, y = create_sequences(scaled_data, df['Price'].values, time_steps=60)
    if len(X) == 0:
        print("Not enough data to create sequences.")
        return
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
    
    print("Building and training the hybrid model (KAN+BiLSTM+MHA)...")
    model = build_hybrid_model((X.shape[1], X.shape[2]))
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,  # Reduced batch size for more stable convergence.
        callbacks=[HideLossCallback(),early_stop, reduce_lr],
        verbose=0
    )
    
    plot_history(history)
    val_loss = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation loss: {val_loss}")
    
    # Predict future price based on data up to the expiration_date.
    expiration_date = pd.to_datetime(expiration_date, format='%Y-%m-%d')
    prediction_df = df[df['Date'] <= expiration_date]
    
    if len(prediction_df) < 60:
        print("Not enough data for prediction.")
        return
    
    last_60_days = prediction_df.tail(60)
    prediction_features = last_60_days[feature_cols]
    input_data = scaler.transform(prediction_features)
    input_data = np.expand_dims(input_data, axis=0)  # (1, 60, n_features)
    
    predicted_price = model.predict(input_data)
    print(f"Predicted price: {predicted_price[0][0]}, Strike price: {strike_price}")
    
    if predicted_price[0][0] > strike_price:
        print("YES")
    else:
        print("NO")
class HideLossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filtered_logs = {k: v for k, v in logs.items() if 'loss' not in k.lower()}  # remove loss entries
        log_strings = [f"{k}: {v:.4f}" for k, v in filtered_logs.items()]
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - " + " - ".join(log_strings))

######################################
# 6. Main Runner
######################################

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python predict.py <csv_file_path> <strike_price> <expiration_date>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    strike_price = float(sys.argv[2])
    expiration_date = sys.argv[3]
    
    main(file_path, strike_price, expiration_date)

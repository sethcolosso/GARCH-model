# ----------------------------------------------
# GARCH(1,1) + Historical Volatility Model
# Forecasting Ex-Ante Volatility (Investor Version)
# ----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf

# ---- 1. Load Data ----
symbol = "NDX"  # Example: NASDAQ 100
data = yf.download(symbol, start="2020-01-01", end="2025-10-01")

# ---- 2. Compute Returns ----
data['Returns'] = 100 * data['Close'].pct_change()
data = data.dropna()

# ---- 3. Compute Historical (Realized) Volatility ----
# Using a 30-day rolling window (standard deviation)
data['HistVol'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)

# ---- 4. Fit GARCH(1,1) ----
model = arch_model(data['Returns'], vol='GARCH', p=1, q=1, dist='normal')
model_fit = model.fit(disp='off')

# ---- 5. Forecast Ex-Ante Volatility ----
forecast = model_fit.forecast(horizon=5)
predicted_vol = np.sqrt(forecast.variance.values[-1, :]) * np.sqrt(252)

# ---- 6. Combine Historical & Forecasted Volatility ----
combined_vol = 0.6 * data['HistVol'].iloc[-1] + 0.4 * predicted_vol.mean()
# (Weights 0.6 historical, 0.4 forecasted — can be adjusted)

print("\n📊 Volatility Overview")
print(f"Last 30-day Historical Volatility: {data['HistVol'].iloc[-1]:.2f}%")
print(f"Average GARCH Ex-Ante Volatility: {predicted_vol.mean():.2f}%")
print(f"Combined Adjusted Volatility (Blended): {combined_vol:.2f}%")

# ---- 7. Visualization ----
plt.figure(figsize=(10, 5))
plt.plot(data.index[-200:], data['HistVol'][-200:], label='Historical Volatility (30D)')
plt.plot(data.index[-200:], model_fit.conditional_volatility[-200:], label='GARCH Conditional Volatility')
plt.axhline(y=combined_vol, color='r', linestyle='--', label='Blended Volatility Forecast')
plt.title(f'{symbol} | Combined Historical + GARCH(1,1) Volatility Forecast')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.show()

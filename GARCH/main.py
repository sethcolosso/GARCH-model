# ----------------------------------------------
# GARCH(1,1) + Historical Volatility Model
# + 90-Day Price Action Chart with MAs + Historical Vol
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
data['HistVol'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)

# ---- 4. Fit GARCH(1,1) ----
model = arch_model(data['Returns'], vol='GARCH', p=1, q=1, dist='normal')
model_fit = model.fit(disp='off')

# ---- 5. Forecast Ex-Ante Volatility ----
forecast_horizon = 5
forecast = model_fit.forecast(horizon=forecast_horizon)
predicted_vol = np.sqrt(forecast.variance.values[-1, :]) * np.sqrt(252)

# ---- 6. Combine Historical & Forecasted Volatility ----
hist_vol_last = data['HistVol'].iloc[-1]
combined_vol = 0.6 * hist_vol_last + 0.4 * predicted_vol.mean()

# ---- 7. Display Summary ----
print("\n📊 Volatility Overview")
print(f"Last 30-day Historical Volatility: {hist_vol_last:.2f}%")
print(f"Average GARCH Ex-Ante Volatility (Next 5 days): {predicted_vol.mean():.2f}%")
print(f"Combined Adjusted Volatility (Blended): {combined_vol:.2f}%")

# ---- 8. Prepare Data for Forecast Plot ----
recent_data = data.iloc[-30:].copy()
recent_dates = list(recent_data.index)
last_date = data.index[-1]
future_dates = pd.bdate_range(last_date, periods=forecast_horizon + 1)[1:]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'PredictedVol': predicted_vol
})

# ---- 9. Plot 1: Past 30D + Forecasted Volatility ----
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax2.plot(recent_data.index, recent_data['Close'], color='gray', linestyle='-', label='NDX Price', alpha=0.5)
ax1.plot(recent_data.index, recent_data['HistVol'], color='blue', label='Historical Volatility (30D)')
ax1.plot(forecast_df['Date'], forecast_df['PredictedVol'], color='orange', label='GARCH(1,1) Forecasted Volatility', marker='o')
plt.axvline(x=last_date, color='red', linestyle='--', label='Forecast Start (Separation Line)')

plt.title(f"{symbol} | Past 30D Historical + Next 5D GARCH(1,1) Volatility Forecast")
ax1.set_xlabel("Date")
ax1.set_ylabel("Volatility (%)", color='blue')
ax2.set_ylabel("NDX Price", color='gray')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# ---- 10. Plot 2: 90-Day Price Action with MAs + Historical Vol ----
price_data = data.iloc[-90:].copy()
price_data['MA15'] = price_data['Close'].rolling(window=15).mean()
price_data['MA30'] = price_data['Close'].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Price and moving averages
ax1.plot(price_data.index, price_data['Close'], label='NDX Price', color='black', linewidth=1.3)
ax1.plot(price_data.index, price_data['MA15'], label='15-Day MA', color='green', linestyle='--', linewidth=1.1)
ax1.plot(price_data.index, price_data['MA30'], label='30-Day MA', color='orange', linestyle='--', linewidth=1.1)

# Volatility on secondary axis
ax2.plot(price_data.index, price_data['HistVol'], color='blue', label='30-Day Historical Volatility', linewidth=1.1)

# Titles and labels
plt.title(f"{symbol} | 90-Day Price Action with 15D & 30D Moving Averages + Historical Volatility")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)", color='black')
ax2.set_ylabel("Volatility (%)", color='blue')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

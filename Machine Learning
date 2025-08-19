# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# === LOAD DATA ===
df_1 = pd.read_csv('Datos_CSV/Sales_Line.csv')
df_1.dropna(inplace=True)

# === CLEANING & FILTERING ===
df_all = df_1[['Quantity', 'Shipment Date', 'Unit Price', 'Unit Cost (LCY)']].copy()
df_all = df_all[(df_all['Quantity'] > 0) & (df_all['Unit Cost (LCY)'] > 0)]
df_all['Shipment Date'] = pd.to_datetime(df_all['Shipment Date'], format='%d-%m-%Y', errors='coerce')
df_all['Year'] = df_all['Shipment Date'].dt.year

# === FILTER YEARS 2016-2022 ===
df = df_all[(df_all['Year'] >= 2016) & (df_all['Year'] <= 2022)].reset_index(drop=True)
df_no_date = df.drop(columns=['Shipment Date'])
df_sum = df_no_date.groupby('Year').sum()

# === PLOTS ===
sns.barplot(x=df_sum.index, y=df_sum['Quantity'])
sns.barplot(x=df_sum.index, y=df_sum['Unit Price'])
sns.barplot(x=df_sum.index, y=df_sum['Unit Cost (LCY)'])

# === SALES COLUMN ===
df['Sales'] = df['Quantity'] * df['Unit Price']
df_sales_sum = df.groupby('Year')[['Quantity', 'Unit Price', 'Unit Cost (LCY)', 'Sales']].sum()
sns.barplot(x=df_sales_sum.index, y=df_sales_sum['Sales'])

# === TIME SERIES FORECASTING (ARIMA) ===
df_per_day = df[['Shipment Date', 'Sales']].groupby('Shipment Date').sum().reset_index()
df_per_day['Shipment Date'] = pd.to_datetime(df_per_day['Shipment Date'])
df_per_day.set_index('Shipment Date', inplace=True)
monthly_sales = df_per_day['Sales'].resample('M').sum().fillna(0)

# ADF Test
result = adfuller(monthly_sales)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Train-Test Split
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales[:train_size], monthly_sales[train_size:]

# ARIMA Forecast
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=24)
forecast_dates = pd.date_range(start=train.index[-1] + pd.DateOffset(months=1), periods=24, freq='M')
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast}).set_index('date')

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test[:24], label='Test')
plt.plot(forecast_df, label='Forecast', linestyle='--')
plt.legend()
plt.title('2-Year Monthly Sales Forecast')
plt.show()

# === ML MODEL FORECASTING ===
yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
X = yearly_sales[['Year']]
y = yearly_sales['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    future_years = pd.DataFrame({'Year': [2021]})
    predicted_sales = model.predict(future_years)
    print(f'Model: {name} → Year: 2021 → Sales: {predicted_sales[0]:,.2f}')

# === FINAL PREDICTIONS FOR FUTURE YEARS ===
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
future_years = pd.DataFrame({'Year': [2023, 2025]})
predicted_sales = dt_model.predict(future_years)
for year, sale in zip(future_years['Year'], predicted_sales):
    print(f'Predicted sales for {year}: {sale:,.2f}')

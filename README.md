# SalesPrediction
This project involves analyzing and forecasting sales data using a real-world dataset containing over 6 million sales records from 2014 to 2021.

---

##Dataset Overview

- **Source**: Internal ERP system
- **Total Rows**: 6,114,141
- **Columns**: 13
- **Fields Used**: 
  - Quantity
  - Shipment Date
  - Unit Price
  - Unit Cost (LCY)

---

##Data Preprocessing

- Removed null values
- Filtered for years **2016 to 2021**
- Removed rows with negative or zero values
- Created features:
  - `Year`
  - `Sales = Quantity × Unit Price`

---

##Exploratory Analysis

- Analyzed trends in quantity sold, price, cost, and total revenue
- Observed:
  - Drop in sales during 2020–2021 (COVID impact)
  - Costs and prices declined accordingly

---

##Forecasting with ARIMA

- Converted to time series: monthly sales
- Verified stationarity using ADF test
- Built ARIMA model to forecast for 2 years ahead
- **Forecasted average monthly sales**: ~€6.21M
- **Annual sales**: ~€74.5M

---

##Machine Learning Models Compared

| Model               | Forecast for 2021 (EUR)  |
|--------------------|--------------------------|
| Linear Regression   | 49.9M                    |
| Ridge Regression    | 53.5M                    |
| Random Forest       | 53.0M                    |
| Gradient Boosting   | 44.9M ✅ Best Match       |
| Support Vector      | 76.4M                    |
| Decision Tree       | 44.9M ✅ Best Match       |

---

##Future Forecast (Decision Tree)

| Year | Predicted Sales (EUR) |
|------|------------------------|
| 2023 | 44.95M                 |
| 2025 | 44.95M                 |

---

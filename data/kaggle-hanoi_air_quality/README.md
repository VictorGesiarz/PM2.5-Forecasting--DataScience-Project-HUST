\# Hanoi PM2.5 Air Quality Dataset with Weather Features



\## Overview

14,451 hourly PM2.5 measurements in Hanoi, Vietnam (Feb 2024 - Jan 2026) with engineered features for forecasting.



\## Quick Stats

\- \*\*Rows\*\*: 14,451

\- \*\*Columns\*\*: 29 features + 1 target

\- \*\*Time Range\*\*: 2024-02-14 to 2026-01-26

\- \*\*Missing Values\*\*: 0%

\- \*\*Mean PM2.5\*\*: 35.92 µg/m³ (above WHO guideline)



\## Features



\### Target Variable

\- `pm25`: PM2.5 concentration (µg/m³)



\### Temporal Features (7)

\- `year`, `month`, `day`, `hour`

\- `day\_of\_week` (0=Monday, 6=Sunday)

\- `is\_weekend` (binary)

\- `is\_dry\_season` (Nov-Apr in Vietnam)



\### Lag Features (7)

⚠️ \*\*Properly implemented - no data leakage\*\*

\- `pm25\_lag1`: PM2.5 from 1 hour ago

\- `pm25\_lag24`: Same hour yesterday

\- `pm25\_lag168`: Same hour last week

\- `pm25\_rolling\_3h`: 3-hour rolling mean

\- `pm25\_rolling\_24h`: 24-hour rolling mean

\- `pm25\_rolling\_7d`: 7-day rolling mean

\- `pm25\_rolling\_24h\_std`: 24-hour volatility



\### Weather Features (15)

\- `temperature`: Air temperature (°C)

\- `humidity`: Relative humidity (%)

\- `dew\_point`: Dew point (°C)

\- `wind\_speed`: Wind speed (m/s)

\- `pressure\_msl`: Sea level pressure (hPa)

\- `precipitation`, `rain`: Precipitation (mm)

\- And more...



\## Sample Usage

```python

import pandas as pd



\# Load data

df = pd.read\_csv('hanoi\_aqi\_ml\_ready\_fixed.csv', 

&nbsp;                parse\_dates=\['datetime'], 

&nbsp;                index\_col='datetime')



\# Quick exploration

print(df.describe())

print(df\['pm25'].hist())



\# Train-test split (time-based)

split\_idx = int(len(df) \* 0.8)

train = df\[:split\_idx]

test = df\[split\_idx:]



\# Simple model

from sklearn.linear\_model import LinearRegression



features = \['pm25\_lag1', 'pm25\_rolling\_24h', 'temperature', 'humidity']

X\_train = train\[features]

y\_train = train\['pm25']



model = LinearRegression()

model.fit(X\_train, y\_train)



\# Expected performance: R² ≈ 0.72, RMSE ≈ 6.9 µg/m³

```



\## Known Issues

✅ \*\*None\*\* - Data is clean and ready to use



\## License

CC0: Public Domain



\## Acknowledgments

\- OpenAQ+WAQI for PM2.5 data

\- Open-Meteo for weather data
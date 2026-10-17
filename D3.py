import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.DataFrame({
    "year": [
        2018, 2020, 2015, 2022, 2019,
        2017, 2021, None, 2016, 2023,
        2014, 2020, 2018, 2015, 2022,
        2019, 2016, 2021, None, 2017,
        2023, 2014, 2019, 2020, 2018,
        2015, 2022, 2017, 2021, 2016,
    ],
    "mileage": [
        80000, 45000, 150000, 20000, 60000,
        110000, 35000, 70000, None, 10000,
        180000, 50000, 75000, 160000, 25000,
        55000, 140000, 30000, 90000, 120000,
        15000, 200000, 65000, None, 85000,
        170000, 22000, 130000, 40000, 145000,
    ],
    "engine_power": [
        150, 190, 120, 250, 160,
        140, 200, 180, 110, 280,
        100, 180, None, 130, 240,
        170, 115, 210, 155, 135,
        300, 95, 165, 185, 145,
        125, 230, 138, None, 118,
    ],
    "price": [
        1800000, 2600000, 900000, 4200000, 2100000,
        1300000, 3100000, 2200000, 750000, 4800000,
        600000, 2700000, 1900000, 850000, 3900000,
        2300000, 800000, 3300000, 2000000, 1100000,
        5100000, 500000, 2200000, 2800000, 1750000,
        700000, 4100000, 1000000, 3200000, 780000,
    ],
})
x = df.drop(columns=["price"])
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
numeric_features = ["year", "mileage", "engine_power"]
imputer_numeric = SimpleImputer(strategy="median")

x_train[numeric_features] = imputer_numeric.fit_transform(
    x_train[numeric_features]
)

x_test[numeric_features] = imputer_numeric.transform(
    x_test[numeric_features]
)

scaler = StandardScaler()

x_train[numeric_features] = scaler.fit_transform(
    x_train[numeric_features]
)

x_test[numeric_features] = scaler.transform(
    x_test[numeric_features]
)

model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

results = pd.DataFrame({
    "real_price": y_test,
    "predicted_price": predictions,
}, index=y_test.index)

print(results)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
print(df)

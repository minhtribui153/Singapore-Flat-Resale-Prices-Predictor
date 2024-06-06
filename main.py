from inquirer import prompt, Text, Checkbox, List
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

resale_flat_prices_df = pd.read_csv("SingaporePublicHousingResaleFlatPrices.csv")

questions = [
    Checkbox(
        "town",
        message="Select towns (or leave blank to select all)",
        choices=resale_flat_prices_df["town"].unique().tolist()
    ),
    Checkbox(
        "flat_type",
        message="Select flat types (or leave blank to select all)",
        choices=resale_flat_prices_df["public_housing_flat_type"].unique().tolist()
    ),
    Checkbox(
        "flat_model",
        message="Select flat models (or leave blank to select all)",
        choices=resale_flat_prices_df["flat_model"].unique().tolist()
    ),
]

print("=====================================================")
print("Prediction for Singapore's public housing flat prices")
print("=====================================================\n")

predict_with = prompt([Checkbox("choice", "What would you like to predict with?", ["Storey Number", "Floor Area"], validate=lambda _, x: len(x) != 0)])["choice"]
storey_num = int(prompt([Text("storey", "Enter storey number", validate=lambda _, x: x.isdigit() and int(x) > 0)])["storey"]) if "Storey Number" in predict_with else None
floor_area = float(prompt([Text("floor area", "Enter floor area", validate=lambda _, x: isfloat(x) and float(x) > 0)])["floor area"]) if "Floor Area" in predict_with else None

answers = prompt(questions)

town = answers["town"] if len(answers["town"]) != 0 else resale_flat_prices_df["town"].unique().tolist()
flat_type = answers["flat_type"] if len(answers["flat_type"]) != 0 else resale_flat_prices_df["public_housing_flat_type"].unique().tolist()
flat_model = answers["flat_model"] if len(answers["flat_model"]) != 0 else resale_flat_prices_df["flat_model"].unique().tolist()

query_df = resale_flat_prices_df[
    resale_flat_prices_df["town"].isin(town) & \
    resale_flat_prices_df["public_housing_flat_type"].isin(flat_type) & \
    resale_flat_prices_df["flat_model"].isin(flat_model)
]

if query_df.shape[0] == 0:
    print("There is no data based on your query for the model to train and predict.")
    exit(0)

columns = ["floor_area_sqm", "max_storey_range"] if storey_num and floor_area else ["floor_area_sqm"] if floor_area else ["max_storey_range"]
X = query_df[columns]
y = query_df[["resale_price_Singapore_dollars"]]

X_dataset = np.array(X).astype("float32")
y_dataset = np.array(y).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regression_model = LinearRegression(fit_intercept=True)

print("[$] Training model...")
regression_model.fit(X_train, y_train)

print("[V] Training model complete")
print("Score: ", regression_model.score(X_test, y_test))
print("Linear Model Coefficient (m): ", regression_model.coef_)
print("Linear Model Coefficient (c): ", regression_model.intercept_)

model_input = pd.DataFrame(
    [[floor_area, storey_num] if floor_area and storey_num else [floor_area] if floor_area else [storey_num]],
    columns=columns
)
resale_price = regression_model.predict(model_input)
print(f"The resale price for the flat in SGD will be: ${resale_price[0][0]:.2f}")
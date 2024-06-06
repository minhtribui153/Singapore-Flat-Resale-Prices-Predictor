import pandas as pd
import numpy as np

resale_flat_prices_df = pd.read_csv("SingaporePublicHousingResaleFlatPrices.csv")

max_storey_range_array = []
for index, value in resale_flat_prices_df["storey_range"].items():
    max_storey_range_array += [int(value[-2:])]

resale_flat_prices_df.insert(5, "max_storey_range", max_storey_range_array, True)
resale_flat_prices_df = resale_flat_prices_df.reset_index(drop=True)
print(resale_flat_prices_df)
resale_flat_prices_df.to_csv("SingaporePublicHousingResaleFlatPricesCleaned.csv")
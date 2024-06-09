import pandas as pd

df = pd.read_csv("./data/credit_card_default_tw.csv")

default_payment_counts = df["default payment next month"].value_counts()

print(default_payment_counts)

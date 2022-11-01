import pandas as pd

path = "./heart.csv"

df = pd.read_csv(path, sep=",")


n = len(df)
n_e = int((60*n)/100)

x = df[df.columns[:-1]]
y = df[df.columns[-1:]]

x_e = x[:n_e]
y_e = y[:n_e]

x_p = x[n_e:]
y_p = y[n_e:]

x_e.to_csv("x_e.csv")
y_e.to_csv("y_e.csv")
x_p.to_csv("x_p.csv")
y_p.to_csv("y_p.csv")


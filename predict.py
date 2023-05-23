from utils import norm_value
import numpy
import pandas

with open("thetas","r") as f:
	thetas = f.read().split(',')

df = pandas.read_csv("./data.csv")
X = df['km'].values
value = float(input("Input value to predict: "))
value = norm_value(numpy.array([value]),  X.max(), X.min())
predicted = value[0] * float(thetas[1]) + float(thetas[0])
print(predicted)

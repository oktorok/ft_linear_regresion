from utils import norm_value
import numpy
import pandas
import config

try:
	with open("thetas","r") as f:
		thetas = f.read().split(',')
except FileNotFoundError as e:
	thetas = [config.theta0, config.theta1]

df = pandas.read_csv(config.DATA_DIR+"/data.csv")
X = df['km'].values
while True:
	try:
		value = float(input("Input value to predict: "))
		break
	except Exception as e:
		print("Introduce a correct value please :/")
		continue
value = norm_value(numpy.array([value]),  X.max(), X.min())
predicted = value[0] * float(thetas[1]) + float(thetas[0])
print(predicted)

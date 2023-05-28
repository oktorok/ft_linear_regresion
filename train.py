import pandas
import numpy
from utils import *
import argparse
import config
import matplotlib.pyplot as plt

def train(X, Y, theta0, theta1, alpha, error_max, iterations):
	estimatePriceVect=numpy.vectorize(estimatePrice)
	length = len(X)
	mileage = X
	price = Y
	#for i in tqdm(range(iterations)):
	while True:
		estimated_prices = estimatePriceVect(mileage, theta0, theta1)
		error = estimated_prices - price
		tmp0 = alpha * 1/length * numpy.sum(error)
		tmp1 = alpha * 1/length * numpy.sum(numpy.multiply(error, mileage))
		if error_max and abs(tmp0) < float(error_max) and abs(tmp1) < float(error_max):
			return theta0, theta1
		theta0 -= tmp0
		theta1 -= tmp1
		if not error_max and not iterations:
			return theta0, theta1
		iterations -= 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Descripción de mi programa')
	parser.add_argument('--iters', type=int, help='Cuantity of iterations in the training model', default=config.iterations)
	parser.add_argument('--graph', action='store_true', help='Add a final graph showing the Regression Model')
	parser.add_argument('--precision', action='store_true', help='Add a final precision medition report')
	parser.add_argument('--extra-precision', action='store_true', help='Show extra precision meditions')
	parser.add_argument('--alpha', type=float, help='Learning Rate', default=config.alpha)
	parser.add_argument('--minerr', type=float, help='Negates --iters, the program continues until the error is less than min_error',default=config.min_error)
	args = parser.parse_args()
	df = pandas.read_csv(config.DATA_DIR+"/data.csv")
	X = df['km'].values
	X_norm = norm_value(X, X.max(), X.min())
	Y = df['price'].values
	theta0 = config.theta0
	theta1 = config.theta1
	alpha = args.alpha
	iterations = args.iterations
	error = args.min_error
	theta0, theta1 = train(X_norm, Y, theta0, theta1,alpha, error, iterations)

	if args.graph:
		y_pred = [theta0 + theta1 * xi for xi in X_norm]
		plt.scatter(X, Y, color='blue', label='Data')
		plt.xlabel('Kilometers')
		plt.ylabel('Price')
		plt.title('ft_linear_regression')
		plt.legend()
		plt.plot(X, y_pred, color='red', label='Regression Line')
		plt.show()
	with open("thetas", "w") as f:
		f.write(f"{theta0},{theta1}")
	if args.precision:
		data = {
			"rae":relative_absolut_error(X_norm, Y, theta0, theta1),
			"mae":mean_absolute_error(X_norm, Y, theta0, theta1),
			"detcoef": determination_coefficient(X_norm, Y, theta0, theta1),
		}
		if args.extra_precision:
			data["rse"] = relative_standard_error(X_norm, Y, theta0, theta1)
			data["mse"]= mean_squared_error(X_norm, Y, theta0, theta1)
		for k in data:
			print(f"{k} = {data[k]}")

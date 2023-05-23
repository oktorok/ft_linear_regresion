import numpy
def norm_value(value, vmax, vmin):
	norm_val = (value-vmin)/(vmax-vmin)
	return norm_val

def unnorm_value(nvalue, vmax, vmin):
	unvalue = (nvalue*(vmax-vmin)) + vmin
	return unvalue


def estimatePrice(mileage, theta0, theta1):
	res = theta0 + mileage * theta1
	return res

def relative_absolut_error(X, Y, theta0, theta1):
	estimated_err = abs(estimatePrice(X, theta0, theta1) - Y)
	mean_desv = abs(Y.mean() - Y)
	rae = numpy.sum(estimated_err) / numpy.sum(mean_desv)
	return rae

def mean_absolute_error(X, Y, theta0, theta1):
	estimated_err = abs(estimatePrice(X, theta0, theta1) - Y)
	mea = numpy.sum(estimated_err)/len(X)
	return mea

def mean_squared_error(X, Y, theta0, theta1):
	estimated_err = estimatePrice(X, theta0, theta1) - Y
	mse = numpy.sum(numpy.dot(estimated_err, estimated_err))/len(X)
	return mse

def relative_standard_error(X, Y, theta0, theta1):
	estimated_err = estimatePrice(X, theta0, theta1) - Y
	mean_desv = Y.mean() - Y
	rse = numpy.sum(numpy.dot(estimated_err, estimated_err)) / numpy.sum(numpy.dot(mean_desv, mean_desv))
	return rse

def determination_coefficient(X, Y, theta0, theta1):
	return (1 - relative_standard_error(X, Y, theta0, theta1)) * 100
	

	

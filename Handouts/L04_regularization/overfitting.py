
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

quad = preprocessing.PolynomialFeatures(degree=2)
cubic = preprocessing.PolynomialFeatures(degree=3)

# The parameters that follow influence the shape of the polynomial (feel free to change)
a = 1.0    # First stationary point of polynomial
b = 1.5    # Second stationary point of polynomial
A = 2.0    # Scaling factor

# Polynomial coefficients (do not change)
theta_true = np.zeros(4)
theta_true[1] = A * a * b
theta_true[2] = - A * (a + b) / 2.0
theta_true[3] = A / 3.0


np.random.seed(42)  # for reproducible results

N = 100
xmin = 0
xmax = 3
x = xmin + (xmax - xmin) * np.random.rand(N)

sigma = 0.05
epsilon = np.random.randn(N) * np.sqrt(sigma)


## Construct the target values
Phi = cubic.fit_transform(x[:, None])
y = Phi @ theta_true + epsilon

# also construct data for plotting later
x_plot = np.linspace(xmin, xmax, 40)
y_true_plot = cubic.fit_transform(x_plot[:, None]) @ theta_true

from sklearn import model_selection

N_train = 10

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, train_size=N_train
)

from sklearn import linear_model, metrics


from sklearn import pipeline
from ipywidgets import widgets


def polyreg_model(degree):
    return pipeline.make_pipeline(
        preprocessing.PolynomialFeatures(degree=degree),
        linear_model.LinearRegression(fit_intercept=False)
    )

def polyreg_demo(degree=3, n_train=10, model_func=polyreg_model):
    # define model
    model = model_func(degree)
    
    X = x[:, None]  # convert to 2d
    test_size = X.shape[0] - n_train
    split = model_selection.train_test_split(X, y, train_size=n_train, test_size=test_size, random_state=12)
    X_train, X_test, y_train, y_test = split
   
    # fit model
    model.fit(X_train, y_train)
    yhat = model.predict(x_plot[:, None])
    
    # compute metrics
    mse_train = metrics.mean_squared_error(y_train, model.predict(X_train))
    mse_test = metrics.mean_squared_error(y_test, model.predict(X_test))

    # make the plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(x_plot, yhat, "k-", lw=3, label="Fitted Model")
    ax.scatter(X_test.flatten(), y_test, color="b", s=60, alpha=0.5, label="Test Data")
    ax.scatter(X_train.flatten(), y_train, color="r", s=80, alpha=0.7, label="Training Data") 
    ax.set_ylim((-20, 20))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')
    ax.set_title((
        "Fitted vs. True Model Responses: "
        f"Ntrain={n_train:d} deg={degree:d} MSEtrain={mse_train:.3f}, "
        f"MSEtest={mse_test:.3f}, NoiseVar={sigma:.3f}"
    ))
    
    return fig





def fit_polyreg_return_mse(train_x, train_y, test_x, test_y, degree):
    model = polyreg_model(degree)
    model.fit(train_x, train_y)
    test_yhat = model.predict(test_x)
    return metrics.mean_squared_error(test_y, test_yhat)


def do_k_fold_validation_polyreg(k, x, y, degrees):
    model_scores = {d: 0.0 for d in degrees}
    
    kf = model_selection.KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(x):  # repeated k times
        # split data
        x_train = x[train_index, None]
        y_train = y[train_index]
        
        x_test = x[test_index, None]
        y_test = y[test_index]
        
        for d in degrees:
            score = fit_polyreg_return_mse(x_train, y_train, x_test, y_test, degree=d)
            model_scores[d] += score
    
    # compute average of model scores
    return {d: mse / k for d, mse in model_scores.items()}


def k_fold_crossval_via_sklearn(k, x, y, degrees):
    scores = {}
    for d in degrees:
        model = polyreg_model(d)
        mses = model_selection.cross_val_score(
            model, 
            x[:, None], 
            y,  
            scoring='neg_mean_squared_error', 
            cv=k
        )
        scores[d] = -mses.mean()
    return scores

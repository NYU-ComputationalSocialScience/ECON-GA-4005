# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def grad_descent(df, x0, epsilon=1e-3, T=200, alpha=0.1):
    """
    Given a gradient function df, staritng starting point x0,
    stopping parameters epsilon and T, and a learning rate alpha;
    find a local minimum of f(x) near x_0 via gradient descent
    """
    x = np.copy(x0)
    trace = []
    for i in range(T):
        df_i = df(x)
        xp = x - alpha * df_i
        err = max(abs(df_i))
        status = {"x": xp, "i": i, "err": err}
        trace.append(status)
        if err < epsilon:
            return trace
        x[:] = xp[:]

    raise ValueError("Failed to converge")


def f(x):
    return -np.exp(-(x[0] ** 2 + x[1] ** 2))


def plot_surf(f, lim=2, **kw):
    n = 400
    x = np.linspace(-lim, lim, n)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # set up 3d plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})

    Z = f([X, Y])

    ax.plot_surface(X, Y, Z, cmap="viridis", **kw)
    return ax


def df(x):
    return -2 * np.asarray(x) * f(x)


def get_trace_xyz(f, trace):
    xy = [i["x"] for i in trace]
    x, y = zip(*xy)
    z = f([np.array(x), np.array(y)])
    return x, y, z


def plot_path(f, trace, **kw):
    ax = plot_surf(f, **kw)
    x, y, z = get_trace_xyz(f, trace)
    ax.scatter3D(x, y, z, c="red")
    ax.plot(x, y, z, c="red")
    ax.view_init(10, 10 * 7)
    return ax


def plot_contour_path(f, trace, lim, ax=None):
    n = 400
    x = np.linspace(-lim, lim, n)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    # set up plot
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)

    x, y, z = get_trace_xyz(f, trace)
    ax.scatter(x, y, c=np.linspace(0.5, 1, len(x)), s=8)
    ax.set_title("Convergence in {} iterations".format(len(x)))
    return ax


def alpha_experiment(alphas):
    N = len(alphas)
    fig, ax = plt.subplots(1, N, figsize=(N * 4, 4))
    for alpha, ax in zip(alphas, ax):
        trace_alpha = grad_descent(df, [2, -0.3], alpha=alpha, T=100_000)
        plot_contour_path(f, trace_alpha, ax=ax)
    fig.tight_layout()
    return fig


def interactive_alpha_experiment(alpha_power, T=50):
    "alpha = 10**alpha_power to make it easy to do things on log10 scale"
    x0 = np.array([2.0, -0.3])
    x = np.copy(x0)
    trace = []

    alpha = 10 ** alpha_power

    for i in range(T):
        df_i = df(x)
        xp = x - alpha * df_i
        err = max(abs(df_i))
        status = {"x": xp, "i": i, "err": err}
        trace.append(status)
        x[:] = xp[:]

    ax = plot_path(f, trace, alpha=0.3)
    ax.set_title(f"alpha = {alpha}")
    return ax


def forward_difference(f, x, delta):
    out = np.zeros_like(x)
    fx = f(x)

    for i in range(len(x)):
        xi = np.copy(x)
        xi[i] += delta
        fx_i = f(xi)
        out[i] = (fx_i - fx) / delta

    return out


def grad_descent_finite_diff(f, x0, delta=1e-4, **kw):
    def df_fd(x):
        return forward_difference(f, x, delta=delta)

    return grad_descent(df_fd, x0, **kw)


def plot_fd_err(f, df, x0):
    x = []
    y = []
    dfdx = df(x0)
    for delta in np.logspace(-15, 0, 70):
        approx_dfdx = forward_difference(f, x0, delta=delta)
        x.append(delta)
        y.append(max(abs(dfdx - approx_dfdx)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(x, y)
    ax.set_xlabel("delta")
    ax.set_ylabel("abs error in ∇f")
    return ax


import tensorflow as tf


def f_tf(x):
    return -tf.exp(-(x[0] ** 2 + x[1] ** 2))


def grad_desc_tf(f, x0, epsilon=1e-3, T=200, alpha=0.1):

    trace = []
    x = tf.Variable(x0)
    for i in range(T):
        with tf.GradientTape() as tape:
            fx = f(x)
        dfdx = tape.gradient(fx, x)
        xp = x - alpha * dfdx
        err = max(abs(dfdx))
        status = dict(
            i=i, fx=fx.numpy(), dfdx=dfdx.numpy(), err=err.numpy(), x=x.numpy()
        )
        trace.append(status)
        if err < epsilon:
            return trace

        x = tf.Variable(xp)

    raise ValueError("No convergence")

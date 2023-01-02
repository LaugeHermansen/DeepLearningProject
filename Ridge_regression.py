#%%

import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm
from matplotlib import pyplot as plt


d = 5
n_train = 1000
n_val = 10000
sigma_omega = 0.5
sigma_epsilon = 5*np.sqrt(10)
n_trials = 10000
optimal_alpha = sigma_epsilon**2 / sigma_omega**2

alphas = np.exp(np.linspace(np.log(optimal_alpha)-5, np.log(optimal_alpha)+5, 50))

errors = []

for trial in tqdm(range(n_trials)):
    # Generate data
    omega = np.random.randn(d)*sigma_omega
    epsilon_train = np.random.randn(n_train)*sigma_epsilon
    epsilon_val = np.random.randn(n_val)*sigma_epsilon
    X_train = np.random.randn(n_train, d)
    X_val = np.random.randn(n_val, d)
    y_train = np.dot(X_train, omega) + epsilon_train
    y_val = np.dot(X_val, omega) + epsilon_val

    mse = []

    for alpha in alphas:
        # Fit model ridge regression
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_val)
        # y_pred = np.dot(X_val, clf.coef_)
        mse.append(np.sum((clf.coef_ - omega)**2))
        # mse.append(np.mean((y_pred - y_val)**2))

    errors.append(mse)

errors = np.array(errors)
errors_mean = np.mean(errors, axis=0)
errors_05 = np.quantile(errors, 0.05, axis=0)
errors_95 = np.quantile(errors, 0.95, axis=0)
errors_median = np.quantile(errors, 0.5, axis=0)

plt.plot(alphas, errors_median, label='median')
plt.plot(alphas, errors_mean, label='mean')
plt.fill_between(alphas, errors_05, errors_95, alpha=0.2)
plt.axvline(optimal_alpha, color='k', linestyle='--')
plt.xscale('log')
plt.xlabel('Regularization strength')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()

#%%

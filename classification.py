from sklearn import linear_model, datasets


def logistic_regression(x, y):
    logreg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
                                             intercept_scaling=1, solver='lbfgs', max_iter=100, multi_class='ovr',
                                             verbose=True, warm_start=True, n_jobs=1)
    t_x = logreg.transform(x)
    logreg.fit(x)
    return logreg


from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model


def get_logistic_regression_model(x, y):
    print 'generating logistic model'
    logreg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
                                             intercept_scaling=1, solver='lbfgs', max_iter=100, multi_class='ovr',
                                             verbose=True)
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    print 'starting the model'
    logreg.fit(t_x, y)
    print 'logistic model created'
    return d, logreg


def apply_logistic_regression_model(x, model):
    d, logreg = model
    t_x = d.transform(x)
    return logreg.predict(t_x)


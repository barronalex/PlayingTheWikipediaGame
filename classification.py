from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, svm


def get_linear_regression_model(x, y):
    print 'generating linear model'
    linreg = linear_model.LinearRegression()
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    linreg.fit(t_x, y)
    return d, linreg


def get_logistic_regression_model_liblinear(x, y):
    print 'generating logistic model, liblinear'
    logreg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
                                             intercept_scaling=1, solver='liblinear', max_iter=5000, multi_class='ovr',
                                             verbose=False)
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    logreg.fit(t_x, y)
    return d, logreg
def get_logistic_regression_model_lbfgs_multinomial(x, y):
    print 'generating logistic model, lbfgs, multinomial'
    logreg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
                                             intercept_scaling=1, solver='lbfgs', max_iter=20000,
                                             multi_class='multinomial', verbose=True)
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    logreg.fit(t_x, y)
    print 'logistic model created'
    return d, logreg


def get_sgd_model_hinge(x, y):
    print 'generating SGD classifier, hinge loss'
    sgd = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                                        n_iter=5, shuffle=True, verbose=False, epsilon=0.1, n_jobs=1, random_state=None,
                                        learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None)
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    sgd.fit(t_x, y)
    return d, sgd
def get_sgd_model_perceptron(x, y):
    print 'generating SGD classifier, perceptron loss'
    sgd = linear_model.SGDClassifier(loss='perceptron', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                                        n_iter=5, shuffle=True, verbose=False, epsilon=0.1, n_jobs=1, random_state=None,
                                        learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None)
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    sgd.fit(t_x, y)
    return d, sgd


def get_svm_model(x, y):
    print 'generating SVM classifier'
    svm_model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                              fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=True,
                              random_state=None, max_iter=500000)
    d = DictVectorizer()
    t_x = d.fit_transform(x)
    svm_model.fit(t_x, y)
    return d, svm_model


def apply_model(x, model):
    d, mod = model
    t_x = d.transform(x)
    return mod.predict(t_x)


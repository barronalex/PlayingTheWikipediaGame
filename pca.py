from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA


def run_pca(examples):
    pca = PCA(n_components=3)
    d = DictVectorizer()
    X = d.fit_transform(examples)
    return pca.fit_transform(X)


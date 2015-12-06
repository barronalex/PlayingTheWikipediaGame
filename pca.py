from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
print "reloading"

def run_pca(examples):
    svd = TruncatedSVD(n_components=3, random_state=42)
    d = DictVectorizer()
    X = d.fit_transform(examples)
    X = csr_matrix.transpose(X)
    print "sparse matrix obtained"
    return svd.fit(X)


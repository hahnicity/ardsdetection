# -*- coding: utf-8 -*-

from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES

# =============================================================================
# Discretisation supervisée
# =============================================================================
# Pas de discretisation par optimisation du chi2 dans les librairies
# cf. http://eric.univ-lyon2.fr/~ricco/cours/slides/discretisation.pdf page 26
# cf. http://orca.st.usm.edu/~zhang/teaching/cos703/cos703notes/DM03_ChiMerge.pdf


class Chi2discretizer:
    """Bin continuous data into ordinal intervals.

    Bottom-up methode. Initial split into n_bins intervals.
    Compute the chi2merges between each adjacent classes.
    Merge the two classes with minimum chi2 value.
    Stop if the minimum chi2merge less than a chi2 threshold at q_chi2 level.

    Read more in the :
    http://eric.univ-lyon2.fr/~ricco/cours/slides/discretisation.pdf
    http://orca.st.usm.edu/~zhang/teaching/cos703/cos703notes/DM03_ChiMerge.pdf

    Parameters
    ----------
    n_bins : int or array-like, shape (n_features,) (default=100)
        To initialize the bottom-up method, the number of initial bins.
        The initial discretizer is the sklearn.preprocessing.KBinsDiscretizer

    strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
        Strategy used to define the widths of the intiial bins.
        The initial discretizer is the sklearn.preprocessing.KBinsDiscretizer

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D k-means
            cluster.

    q_chi2 : float
            lower tail probability for chi-square threshold
            See scipy.stats.chi2.ppf

    Attributes
    ----------
    n_init_bins_ : int array, shape (n_features,)
        number of initial bins


    df_bin_edges_ : dataframe of arrays (1,n_features)
        The edges of each bin of each feature.
        For each feature, contain arrays of varying shapes ``(n_bins_, )``

    df_n_bins : dataframe (1, n_features)
        number of final bins for each feature

    q_chi2_ : float
        lower tail probability for chi-square threshold

    n_features_ : integer
        number of feature fitted

    See also
    --------
     sklearn.preprocessing.KBinsDiscretizer
    """

    def __init__(self, n_bins=100, strategy='quantile', q_chi2=0.95):
        self.n_init_bins_ = n_bins
        self.init_strategy_ = strategy
        self.q_chi2_ = q_chi2

    def fit(self, X, y):
        """Fits the estimator.

        Parameters
        ----------
        X : numeric pandas.DataFrame, shape (n_samples, n_features)
            Data to be discretized.

        y : numeric pandas.Series, shape (n_samples,)
            target binary variable

        Returns
        -------
        self
        """

        self.df_bin_edges_ = pd.DataFrame(columns=X.columns)
        self.df_n_bins_ = pd.DataFrame(columns=X.columns)
        for i in range(X.shape[1]):

            # Inital binarization
            x = X.iloc[:,i]
            kb = KBinsDiscretizer(n_bins=self.n_init_bins_, strategy=self.init_strategy_, encode='ordinal')
            x_discret = kb.fit_transform(x.values.reshape(len(x),1))[:,0]
            bin_edges = kb.bin_edges_[0] # toutes les bornes sup des intervalles (inférieur ou égale)

            # Matrice contingence
            obs = pd.crosstab(index=x_discret, columns=y)

            # Calcul des chi2merge : comparaison d'une ligne avec l'autre dans obs
            # si deux lignes ne sont pas indépendantes selon un test du chi2, alors on les regroupe
            # On regroupe d'abord les 2 lignes les plus similaiers avant de recalculer les chi2merge
            # cf. http://orca.st.usm.edu/~zhang/teaching/cos703/cos703notes/DM03_ChiMerge.pdf
            threshold = stats.chi2.ppf(self.q_chi2_, df=1) # 2 lignes comparées, et 2 classes (def ou non), donc dll=1
            chi2s = chi2merge(obs) # liste des chi2merge
            while min(chi2s) < threshold:
                row  = chi2s.index(min(chi2s))
                obs.iloc[row] += obs.iloc[row+1]
                obs = obs.drop(axis=1, index=obs.index[row+1])
                chi2s = chi2merge(obs)
                if len(obs) < 2: break

            # Calcul des bornes des classes
            bin_edges = [kb.bin_edges_[0][int(i)] for i in obs.index]

            # Save
            self.df_bin_edges_[x.name] = [bin_edges]
            self.df_n_bins_[x.name] = [len(bin_edges)]
            self.n_features_ = self.df_n_bins_.shape[1]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, X):
        """Discretizes the data.

        Parameters
        ----------
        X : numeric pandas.DataFrame, shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        X_discret : numeric pandas.DataFrame
            Data in the binned space.
        """
        check_is_fitted(self, ["df_bin_edges_"])

        check_array(X, dtype=FLOAT_DTYPES)
        X_discret = X.copy()
        if X.shape[1] != self.n_features_:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(self.n_features_, X.shape[1]))

        for i in range(X.shape[1]):
            x = X.iloc[:,i]
            if self.df_n_bins_.loc[0, x.name] > 1:
                X_discret.iloc[:,i] = applyEdges(self.df_bin_edges_.iloc[0,i], x)

        return X_discret


def chi2merge(obs):
# obs est un DataFrame avec la matrice de contingence.
# Calcul des chi2 entre deux lignes (chi2merge)
# Return la liste des chi2merge
    chi2merge = []
    for i in range(len(obs)-1):
    # Pour calculer le chi2 même en cas de valeur de nulle dans la matrice expected:
    # calcul manuel en ignorant les NaN
        chi2 = np.nansum((obs.iloc[i:i+2] - stats.contingency.expected_freq(obs.iloc[i:i+2]))**2/stats.contingency.expected_freq(obs.iloc[i:i+2]))
        chi2merge.append( chi2)

    return chi2merge

def applyEdges(bin_edges, x):
# x variable continue as pandas.Series
# Discretisation de X avec les bornes définies par bin_dges
# x_discret=i si x in ] bin_edges[i-1]: bin_edges[i] ]
# return x_discretized
    x_discret = np.zeros(len(x))
    for i in range(len(x)):
        j=0
        while (x.iloc[i] > bin_edges[j]) & (j<len(bin_edges)-1):
            j+=1
        x_discret[i] = j
    return x_discret

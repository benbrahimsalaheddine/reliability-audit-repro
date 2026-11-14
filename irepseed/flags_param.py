# FaeU W1 / F-W1.3, F-W1.4: per-panel flags of tau_p^2 under equal loadings and under free loadings, on every dataset with >= 3 panels. Seed 0. 12 Nov 2026.
# Equal-loading estimator (as in v64): tau_hat_p = Var(ybar_p) - mean_{q != p} Cov(ybar_p, ybar_q) - noise_p.
# Free-loading estimator: tau_hat_p = Var(ybar_p) - noise_p - a_p^2, with a_p^2 = S_pq S_pr / S_qr (P = 3, triad identity) or from the one-factor
#   least-squares fit of the off-diagonals (P >= 4).  noise_p = mean within-item variance (ddof=1) * mean(1/k), as in the v64 IRep script.
# A panel is flagged when the 5th percentile of 300 item-bootstrap draws is above 0 (one-sided 5%).  Items: >= 2 ratings in every panel.
import warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares
warnings.filterwarnings('ignore')
rng = np.random.default_rng(int(__import__("sys").argv[1])); B = int(__import__("sys").argv[2])

def a2_fit(S):
    P = S.shape[0]
    if P == 3:
        return np.array([S[0, 1] * S[0, 2] / S[1, 2], S[0, 1] * S[1, 2] / S[0, 2], S[0, 2] * S[1, 2] / S[0, 1]])
    iu = np.triu_indices(P, 1)
    a = least_squares(lambda a: np.outer(a, a)[iu] - S[iu], np.sqrt(np.abs(S[iu]).mean()) * np.ones(P)).x
    return a ** 2

def taus(X, noise):
    P = X.shape[1]; S = np.cov(X.T, ddof=1)
    eq = np.array([S[p, p] - np.mean([S[p, q] for q in range(P) if q != p]) - noise[p] for p in range(P)])
    fr = np.diag(S) - noise - a2_fit(S)
    return eq, fr

def run(name, X, noise, names, quiet=False):
    n = X.shape[0]; eq, fr = taus(X, noise)
    be, bf = [], []
    for _ in range(B):
        try:
            e, f = taus(X[rng.integers(0, n, n)], noise); be.append(e); bf.append(f)
        except Exception: pass
    qe, qf = np.percentile(be, 5, axis=0), np.percentile(bf, 5, axis=0)
    if not quiet:
        print('===== %s | panels %d | items %d' % (name, len(names), n))
        for i, nm in enumerate(names):
            print('   %-16s equal: tau_hat %+.4f (5%% q %+.4f)%s | free: tau_hat %+.4f (5%% q %+.4f)%s' % (str(nm)[:16], eq[i], qe[i], ' FLAG' if qe[i] > 0 else '     ', fr[i], qf[i], ' FLAG' if qf[i] > 0 else ''))
        print('   flagged panels: equal %d of %d, free %d of %d' % ((qe > 0).sum(), len(names), (qf > 0).sum(), len(names)))
    return qe > 0, qf > 0

def from_long(D):
    g = D.groupby(['item', 'panel'], sort=False).y
    K = g.count().unstack(); ok = (K >= 2).all(axis=1)
    M = g.mean().unstack()[ok]; V = g.var(ddof=1).unstack()[ok]; Kk = K[ok]
    return M.values, (V.mean() * (1 / Kk).mean()).values, list(M.columns)

R = pd.read_csv('./data/d3repo/dataset/d3-ratings.csv'); H = pd.read_csv('./data/d3repo/dataset/d3-raters.csv')
R = R.merge(H[['rater_id', 'Region']], on='rater_id'); R = R[R.rating_raw >= 0]
X, nz, nm = from_long(R.rename(columns={'item_id': 'item', 'Region': 'panel', 'rating_raw': 'y'})[['item', 'panel', 'y']]); run('D3code', X, nz, nm)
C = pd.read_csv('./data/crehate/data/CREHate_raw_annotations.csv')
X, nz, nm = from_long(C.rename(columns={'ID': 'item', 'nationality': 'panel', 'annotation': 'y'})[['item', 'panel', 'y']]); run('CREHate', X, nz, nm)
J = pd.read_csv('/tmp/jig/specialized_rater_pools_data.csv', usecols=['id', 'toxic_score', 'rater_group']).dropna(subset=['toxic_score'])
X, nz, nm = from_long(J.rename(columns={'id': 'item', 'rater_group': 'panel', 'toxic_score': 'y'})); run('Jigsaw', X, nz, nm)

d = pd.read_csv('/tmp/irep/replication-dataset/IREP_data_release.csv'); d.columns = [c.strip('"') for c in d.columns]
lab = [c for c in d.columns if c not in ('item_id', 'platform', 'rater', 'error.other')]
pools = ['Budapest', 'Mexico', 'KualaLumpur']
s = d[d.platform.isin(pools)]; cnt = s.groupby(['item_id', 'platform']).size().unstack(); items = cnt.index[(cnt[pools] >= 2).all(axis=1)]; s = s[s.item_id.isin(items)]
fe = np.zeros(3, int); ff = np.zeros(3, int); used = 0
for c in lab:
    g = s.groupby(['item_id', 'platform'])[c]; X = g.mean().unstack()[pools].values; V = g.var(ddof=1).unstack()[pools]; K = g.count().unstack()[pools]
    noise = V.mean().values * (1 / K).mean().values
    S = np.cov(X.T, ddof=1)
    if not np.all(S[np.triu_indices(3, 1)] > 0): continue
    e, f = run(c, X, noise, pools, quiet=True); fe += e; ff += f; used += 1
print('===== IRep | 3 pools | items %d | labels used %d of %d (all three off-diagonal covariances > 0)' % (len(items), used, len(lab)))
print('   labels flagged by pool, equal loadings:', dict(zip(pools, fe.tolist())))
print('   labels flagged by pool, free loadings :', dict(zip(pools, ff.tolist())))
print('   (v64 reports with equal loadings over all 31 labels: Kuala Lumpur 25, Budapest 15, Mexico 8)')

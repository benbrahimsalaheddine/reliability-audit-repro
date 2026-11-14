# Reproduces the "LLM judges as panels" numbers from public data only.
# Data: lguerdan/indeterminacy-experiments (runs/main-run/summ_eval_*), SummEval model_annotations.aligned.jsonl
# Usage: python3 judges_as_panels_v121.py DATA_DIR   (DATA_DIR has the four dimension folders and the jsonl)
import json, sys, csv, re
import numpy as np
rng_seed = 0
D = sys.argv[1] if len(sys.argv) > 1 else './data/summeval'
dims = ['coherence', 'consistency', 'fluency', 'relevance']
judges = ['anthropic-claude-3-5-sonnet-20241022', 'anthropic-claude-3-haiku-20240307', 'deepseek-deepseek-chat',
          'llama-Llama-3.3-70B-Instruct', 'mistral-mistral-large-latest', 'mistral-mistral-small-latest',
          'openai-gpt-3.5-turbo', 'openai-gpt-4o-mini-2024-07-18', 'openai-o3-mini']
norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower())
ann = {}
for line in open(f'{D}/model_annotations.aligned.jsonl'):
    r = json.loads(line)
    ann.setdefault((r['id'], norm(r['decoded'])), r)  # first record in file order for duplicated texts
csv.field_size_limit(10**9)
out = {}
for d in dims:
    rows = list(csv.DictReader(open(f'{D}/{d}/ratings.csv')))
    keys = [(r['id'], norm(r['summary'])) for r in rows]
    miss = [k for k in keys if k not in ann]
    assert not miss, (d, len(miss))
    E = np.array([np.mean([a[d] for a in ann[k]['expert_annotations']]) for k in keys])
    C = np.array([np.mean([a[d] for a in ann[k]['turker_annotations']]) for k in keys])
    S = np.zeros((len(rows), len(judges))); nvalid = np.ones((len(rows), len(judges)), bool)
    for j, jn in enumerate(judges):
        rt = np.array(json.load(open(f'{D}/{d}/{jn}.json'))['resp_table'], float)  # item, FC option, RS, draw
        pos = rt[:, 0].sum((1, 2)); tot = rt.sum((1, 2, 3))
        ok = tot > 0
        S[:, j] = np.where(ok, pos / np.where(ok, tot, 1), np.nan)
    out[d] = (E, C, S)
res = {}
print('dim | mean pairwise judge corr | first-PC share | corr(first PC, expert) | corr(first PC, crowd) | R2 exp beyond crowd | R2 crowd beyond exp')
rng = np.random.default_rng(rng_seed)
tot_pairs = 0; excl90 = 0; excl95 = 0
for d in dims:
    E, C, S = out[d]
    keep = ~np.isnan(S).any(1)
    Ek, Ck, Sk = E[keep], C[keep], S[keep]
    R = np.corrcoef(Sk.T)
    off = R[np.triu_indices(9, 1)]
    w, v = np.linalg.eigh(R)
    share = w[-1] / 9
    z = (Sk - Sk.mean(0)) / Sk.std(0)
    f = z @ v[:, -1]
    if np.corrcoef(f, z.mean(1))[0, 1] < 0: f = -f
    ce, cc = np.corrcoef(f, Ek)[0, 1], np.corrcoef(f, Ck)[0, 1]
    m = Sk.mean(1)
    def r2(y, X):
        X = np.column_stack([np.ones(len(y))] + X); b = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r_c, r_e, r_ce = r2(m, [Ck]), r2(m, [Ek]), r2(m, [Ek, Ck])
    print(f'{d:11s} n={keep.sum()} | {off.mean():.2f} ({off.min():.2f}-{off.max():.2f}) | {share:.3f} | {ce:.2f} | {cc:.2f} | {r_ce - r_c:.2f} | {r_ce - r_e:.2f}')
    # per judge expert-crowd gap with 1000 item draws, Pearson
    idx = rng.integers(0, len(Ek), size=(1000, len(Ek)))
    for j in range(9):
        s = Sk[:, j]
        gap = np.corrcoef(s, Ek)[0, 1] - np.corrcoef(s, Ck)[0, 1]
        gb = np.empty(1000)
        for b in range(1000):
            i = idx[b]; gb[b] = np.corrcoef(s[i], Ek[i])[0, 1] - np.corrcoef(s[i], Ck[i])[0, 1]
        lo90, hi90 = np.percentile(gb, [5, 95]); lo95, hi95 = np.percentile(gb, [2.5, 97.5])
        tot_pairs += 1; excl90 += (lo90 > 0 or hi90 < 0); excl95 += (lo95 > 0 or hi95 < 0)
        res[(d, j)] = gap
print('gap excludes zero: 90% interval', excl90, 'of', tot_pairs, '| 95% interval', excl95, 'of', tot_pairs)
for d in dims:
    g = np.array([res[(d, j)] for j in range(9)])
    print(d, 'mean gap %.2f' % g.mean(), 'n gap>0.35:', int((g > 0.35).sum()), 'n gap>0:', int((g > 0).sum()))

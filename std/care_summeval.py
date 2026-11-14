# CARE-SVD (authors' own code, github.com/SprocketLab/CARE, commit cloned 12 Nov 2026) applied to SummEval.
# We load ONLY their functions learn_structure and get_weights, unmodified, via ast (the rest of pgm_tools needs snorkel/torch, which we do not use).
# Columns of the judge matrix X (items x raters), per SummEval dimension:
#   A) 3 panel means: J (9 LLM judges, share of positive answers), E (3 experts), C (5 crowd workers)
#   B) 17 raters: 9 judges + 3 expert slots + 5 crowd slots (crowd slots are exchangeable, not persistent persons)
# CARE input = correlation matrix of X. gamma is the sparsity weight; the authors tune it on a validation set against a ground truth.
# We have no ground truth, so we report several gamma values from their grid. SCS solver, max_iters=10000 as in their script. No randomness: seed not used.
import ast, sys, json, csv, re, numpy as np, pandas as pd, scipy as sp, cvxpy as cp
csv.field_size_limit(10**9)
src = open('/tmp/care_repo/src/src/pgm_tools.py').read(); tree = ast.parse(src)
ns = {'cp': cp, 'np': np, 'sp': sp, 'pd': pd}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('learn_structure', 'get_weights'):
        exec(compile(ast.Module([node], []), 'pgm_tools', 'exec'), ns)
learn_structure, get_weights = ns['learn_structure'], ns['get_weights']
D = './data/summeval'
judges = ['anthropic-claude-3-5-sonnet-20241022', 'anthropic-claude-3-haiku-20240307', 'deepseek-deepseek-chat', 'llama-Llama-3.3-70B-Instruct', 'mistral-mistral-large-latest', 'mistral-mistral-small-latest', 'openai-gpt-3.5-turbo', 'openai-gpt-4o-mini-2024-07-18', 'openai-o3-mini']
norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower())
ann = {}
for l in open(f'{D}/model_annotations.aligned.jsonl'):
    r = json.loads(l); ann.setdefault((r['id'], norm(r['decoded'])), r)
def load(d):
    rows = list(csv.DictReader(open(f'{D}/{d}/ratings.csv'))); keys = [(r['id'], norm(r['summary'])) for r in rows]
    S = []
    for jn in judges:
        rt = np.array(json.load(open(f'{D}/{d}/{jn}.json'))['resp_table'], float); pos = rt[:, 0].sum((1, 2)); n = rt.sum((1, 2, 3))
        S.append(np.where(n > 0, pos / np.where(n > 0, n, 1), np.nan))
    S = np.array(S).T; ok = ~np.isnan(S).any(1); S = S[ok]; kk = [k for k, o in zip(keys, ok) if o]
    Er = np.array([[a[d] for a in ann[k]['expert_annotations']] for k in kk], float)
    Cr = np.array([[a[d] for a in ann[k]['turker_annotations']] for k in kk], float)
    return S, Er, Cr
def run(X, names, groups, gamma):
    corr = pd.DataFrame(X, columns=names).corr().values
    Sx, Lx = learn_structure(corr, gamma=gamma, solver=cp.SCS, verbose=False, max_iters=10000)
    ev, evec = np.linalg.eigh(0.5 * (Lx + Lx.T)); idx = np.argsort(ev)[::-1]; ev, evec = ev[idx], evec[:, idx]
    w = get_weights(Lx)
    out = {'eig': [float(round(e, 4)) for e in ev[:3]]}
    for f in (0, 1):
        v = evec[:, f] * np.sqrt(max(ev[f], 0)); v = v if v[np.argmax(np.abs(v))] > 0 else -v
        out[f'factor{f+1}'] = {g: float(np.round(np.mean([v[i] for i in range(len(names)) if groups[i] == g]), 3)) for g in sorted(set(groups))}
    wn = w / np.sum(w) if abs(np.sum(w)) > 1e-12 else w
    out['aggregation_weight_share'] = {g: float(np.round(np.sum([wn[i] for i in range(len(names)) if groups[i] == g]), 3)) for g in sorted(set(groups))}
    return out
for d in ['coherence', 'consistency', 'fluency', 'relevance']:
    S, Er, Cr = load(d); n = len(S)
    print(f'=== {d}: n={n} items', flush=True)
    XA = np.column_stack([S.mean(1), Er.mean(1), Cr.mean(1)])
    XB = np.column_stack([S, Er, Cr]); namesB = [f'J{i}' for i in range(9)] + [f'E{i}' for i in range(3)] + [f'C{i}' for i in range(5)]; grpB = ['J'] * 9 + ['E'] * 3 + ['C'] * 5
    for gamma in (0.5, 1, 3):
        print(f' A) 3 panel means, gamma={gamma}: {run(XA, ["J", "E", "C"], ["J", "E", "C"], gamma)}', flush=True)
        print(f' B) 17 raters,     gamma={gamma}: {run(XB, namesB, grpB, gamma)}', flush=True)

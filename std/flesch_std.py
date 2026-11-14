# Flesch Reading Ease and Flesch-Kincaid grade with a simple vowel-group syllable counter (no external dictionary).
import json,re,sys
def syl(w):
    w=w.lower(); w=re.sub(r"[^a-z]","",w)
    if not w: return 0
    n=len(re.findall(r"[aeiouy]+",w))
    if w.endswith('e') and not w.endswith(('le','ee')) and n>1: n-=1
    return max(n,1)
def sents(t): return [s for s in re.split(r"(?<=[.!?])\s+",t) if len(s.split())>=3]
def scores(t):
    S=sents(t); W=[w for w in re.findall(r"[A-Za-z][A-Za-z'-]*",t)]
    ns,nw=len(S),len(W); sy=sum(syl(w) for w in W)
    fre=206.835-1.015*nw/ns-84.6*sy/nw; fk=0.39*nw/ns+11.8*sy/nw-15.59
    return nw,ns,fre,fk
def clean(p):
    p=re.sub(r"\([^)]*\d{4}[^)]*\)","",p)
    p=re.sub(r"\S*[σρτμℓȳĉ√≈≤≥∣→⋆²₀-₉ₐ-ₜ=±]\S*","",p)
    p=re.sub(r"\[[^\]]*\]|\bFig\.?|\bProp\.?|\bApp\.?","",p)
    return re.sub(r"\s+"," ",p).strip()
if __name__=="__main__":
    S=json.load(open('/tmp/secs.json'))
    for k,pp in S.items():
        t=' '.join(clean(p) for p in pp if len(p.split())>=8 and not p.startswith(('Figure','Table','|')))
        nw,ns,fre,fk=scores(t); print(f"{k:6s} words {nw:5d} sentences {ns:4d}  words/sentence {nw/ns:5.1f}  Flesch Reading Ease {fre:5.1f}  Flesch-Kincaid grade {fk:5.1f}")

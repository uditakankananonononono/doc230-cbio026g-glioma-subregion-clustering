import numpy as np, nibabel as nib, json, glob, os, time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_rand_score
D='/tmp/brats/BraTS24-GLI/'
rng=np.random.default_rng(0)
cases=sorted(set(os.path.basename(f)[:-11] for f in glob.glob(D+'Masks/*-seg.nii.gz')))
assert len(cases)==40, len(cases)
mods=['t1n','t1c','t2w','t2f']
X=[];Y=[];P=[];XR=[];PA=[]
t0=time.time()
offs=[(i,j) for i in range(-2,3) for j in range(-2,3)]
for pi,c in enumerate(cases):
    vols=[nib.load(f"{D}Images-{m}/{c}-{m}.nii.gz").get_fdata(dtype=np.float32) for m in mods]
    seg=np.rint(nib.load(f"{D}Masks/{c}-seg.nii.gz").get_fdata()).astype(np.int8)
    bm=np.all([v!=0 for v in vols],0)
    idx=np.argwhere(bm); sel=idx[rng.choice(len(idx),20000,replace=False)]
    raw=np.stack([v[tuple(sel.T)] for v in vols],1)
    z=[(v-v[bm].mean())/(v[bm].std()+1e-6) for v in vols]
    zz=np.stack([q[tuple(sel.T)] for q in z],1)
    sh=vols[0].shape; pats=[]
    for zq in z:
        for (i,j) in offs:   # edge-clipped patch coordinates
            a=np.clip(sel[:,0]+i,0,sh[0]-1); b=np.clip(sel[:,1]+j,0,sh[1]-1)
            pats.append(zq[a,b,sel[:,2]])
    PA.append(np.stack(pats,1).astype(np.float32))
    X.append(zz);XR.append(raw);Y.append(seg[tuple(sel.T)]);P.append(np.full(20000,pi))
    print(pi,c,round(time.time()-t0),flush=True)
X=np.concatenate(X);XR=np.concatenate(XR);Y=np.concatenate(Y).astype(int);P=np.concatenate(P);PA=np.concatenate(PA)
print('label frac',np.bincount(Y)/len(Y),flush=True)
LAB=[3,1,2,4]; NAME={3:'ET',1:'NETC',2:'SNFH',4:'RC'}
def dice_l(yt,yp,l): a=yt==l;b=yp==l; return float(2*(a&b).sum()/(a.sum()+b.sum()+1e-9))
def macro(yt,yp): return float(np.mean([dice_l(yt,yp,l) for l in LAB]))
def kfold():
    f=P%5
    for k in range(5): yield k,np.where(f!=k)[0],np.where(f==k)[0]
def cluster(Xf):
    ari=[];sh=[];preds=np.zeros(len(Y),int)
    for k,tr,te in kfold():
        km=KMeans(5,n_init=3,random_state=0).fit(Xf[tr]); ct=km.predict(Xf[tr]); cte=km.predict(Xf[te])
        mp={c:np.bincount(Y[tr][ct==c],minlength=5).argmax() for c in range(5)}
        preds[te]=[mp[c] for c in cte]
        ari.append(adjusted_rand_score(Y[te],cte))
        yp=Y[te].copy()
        for p in np.unique(P[te]):
            m=P[te]==p; yp[m]=rng.permutation(yp[m])
        sh.append(adjusted_rand_score(yp,cte))
    return ari,sh,preds

R={'cases':cases}
srng=np.random.default_rng(0)
def fit_rf(F,cw=None):
    pr=np.zeros(len(Y),int)
    for k,tr,te in kfold():
        tr=srng.choice(tr,200000,replace=False)
        pr[te]=RandomForestClassifier(30,max_depth=12,n_jobs=2,random_state=0,class_weight=cw).fit(F[tr],Y[tr]).predict(F[te])
    return pr
def rep(pr): return dict(dice={NAME[l]:dice_l(Y,pr,l) for l in LAB},macro=macro(Y,pr))
pr_ref=fit_rf(X); R['ref_RF']=rep(pr_ref); print('ref',R['ref_RF'],flush=True)
# K6
vals=[];n=0
for p in np.unique(P):
    m=P==p
    if (Y[m]==3).sum()>=50: vals.append(dice_l(Y[m],pr_ref[m],3))
R['K6']=dict(n_patients=len(vals),median_ET_dice=float(np.median(vals)),q25=float(np.percentile(vals,25)),q75=float(np.percentile(vals,75)),gate_met=bool(np.median(vals)>0.20)); print('K6',R['K6'],flush=True)
# K7
R['K7']={}
for j,mn in enumerate(['t1n','t1c','t2w','t2f']):
    keep=[i for i in range(4) if i!=j]; pr=fit_rf(X[:,keep]); r=rep(pr); R['K7'][mn]=r; print('K7',mn,r,flush=True)
R['K7']['gate_met']=bool(R['ref_RF']['dice']['ET']-R['K7']['t1c']['dice']['ET']>=0.03)
# K8
R['K8']={}
for kk in (3,5,8):
    ar=[]
    for k,tr,te in kfold():
        km=KMeans(kk,n_init=3,random_state=0).fit(X[tr]); ar.append(adjusted_rand_score(Y[te],km.predict(X[te])))
    R['K8'][str(kk)]=float(np.mean(ar))
R['K8']['gate_met']=bool(max(R['K8'][str(k)] for k in (3,5,8))>0.10); print('K8',R['K8'],flush=True)
# K9
pa=PA.reshape(len(PA),4,25); CF=np.hstack([X,pa.mean(2),pa.std(2)]); pr9=fit_rf(CF); R['K9']=rep(pr9); R['K9']['gate_met']=bool(R['K9']['macro']>R['ref_RF']['macro']+0.02); print('K9',R['K9'],flush=True)
# K10
pr10=fit_rf(X,'balanced_subsample'); R['K10']=rep(pr10); R['K10']['gate_met']=bool(R['K10']['macro']>R['ref_RF']['macro']+0.02); print('K10',R['K10'],flush=True)
json.dump(R,open('results/ext_results.json','w'),indent=1); print('done',round(time.time()-t0))

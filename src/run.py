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
ari,sh,pr1=cluster(X)
R['K1']=dict(ari_folds=ari,ari=float(np.mean(ari)),shuffle_ari=float(np.mean(sh)),gate='ARI>0.10 and > shuffle+0.05'); R['K1']['met']=bool(R['K1']['ari']>0.10 and R['K1']['ari']>R['K1']['shuffle_ari']+0.05)
R['K2']=dict(dice={NAME[l]:dice_l(Y,pr1,l) for l in LAB},macro=macro(Y,pr1),gate='ET Dice>0.30'); R['K2']['met']=bool(R['K2']['dice']['ET']>0.30)
print('K1',R['K1'],'K2',R['K2'],flush=True)
srng=np.random.default_rng(0)
def sup(model_fn,F,shuffle=False):
    pr=np.zeros(len(Y),int)
    for k,tr,te in kfold():
        tr=srng.choice(tr,200000,replace=False)
        yt=Y[tr].copy()
        if shuffle:
            for p in np.unique(P[tr]):
                m=P[tr]==p; yt[m]=srng.permutation(yt[m])
        pr[te]=model_fn().fit(F[tr],yt).predict(F[te]); print(' fold',k,round(time.time()-t0),flush=True)
    return pr
rf=lambda:RandomForestClassifier(30,max_depth=12,n_jobs=2,random_state=0)
pr3=sup(rf,X); R['K3']=dict(dice={NAME[l]:dice_l(Y,pr3,l) for l in LAB},macro=macro(Y,pr3),gate='macro Dice > K2 macro')
R['K3']['met']=bool(R['K3']['macro']>R['K2']['macro'])
pr3s=sup(rf,X,True); R['K3']['shuffle_macro']=macro(Y,pr3s)
print('K3',R['K3'],flush=True)
mlp=lambda:MLPClassifier((64,32),max_iter=60,early_stopping=True,random_state=0)
pr4=sup(mlp,PA); R['K4']=dict(dice={NAME[l]:dice_l(Y,pr4,l) for l in LAB},macro=macro(Y,pr4),gate='macro Dice > K3+0.02')
R['K4']['met']=bool(R['K4']['macro']>R['K3']['macro']+0.02)
print('K4',R['K4'],flush=True)
ari5,sh5,_=cluster(XR); R['K5']=dict(ari_raw=float(np.mean(ari5)),ari_z=R['K1']['ari'],delta=float(np.mean(ari5)-R['K1']['ari']),gate='|delta|<0.05'); R['K5']['met']=bool(abs(R['K5']['delta'])<0.05)
print('K5',R['K5'],flush=True)
R['label_frac']=(np.bincount(Y)/len(Y)).tolist()
json.dump(R,open('results/results.json','w'),indent=1)
print('done',time.time()-t0)

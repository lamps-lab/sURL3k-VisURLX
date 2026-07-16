import os, glob, argparse, collections
import pandas as pd
import exact_url_evaluator as E

FINE={0:"general-url",1:"third-party-dataset",2:"author-provided-dataset",
      3:"third-party-software",4:"author-provided-software",5:"project"}
N2ID={v:k for k,v in FINE.items()}
GRAN={
 "fine":   ({i:FINE[i] for i in range(6)},
            [FINE[i] for i in range(6)]),
 "coarse": ({0:"general-url",1:"dataset",2:"dataset",3:"software",4:"software",5:"project"},
            ["general-url","dataset","software","project"]),
 "binary": ({0:"not-OADS",1:"OADS",2:"OADS",3:"OADS",4:"OADS",5:"OADS"},
            ["OADS","not-OADS"]),
}
def majority(xs): return collections.Counter(xs).most_common(1)[0][0]
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    return P,R,(2*P*R/(P+R) if P+R else 0.0)

def load_gold(gold_dir):
    pu=collections.defaultdict(collections.Counter); rows=[]
    for f in glob.glob(os.path.join(gold_dir,"*-gold.csv")):
        d=pd.read_csv(f,dtype=str,keep_default_na=False)
        for _,r in d.iterrows():
            if str(r.get("split","")).strip()!="test" or str(r.get("Label","")).strip()=="": continue
            c=int(float(r["Label"].split('.')[0])); pid=E._normalize_paper_id(r["paper_id"])
            rows.append(c)
            for u in E._extract_norm_urls_from_field(r["url"]): pu[(pid,u)][c]+=1
    gold_pu={k:v.most_common(1)[0][0] for k,v in pu.items()}
    return gold_pu, rows

def discover(pred_dirs):
    out={}
    for pd_ in pred_dirs:
        for f in sorted(glob.glob(os.path.join(pd_,"*_all_predictions.csv"))):
            b=os.path.basename(f)
            if b.endswith("_gold_all_predictions.csv"): continue
            out[b[:-len("_all_predictions.csv")]]=f
    return out

def score(path, gran, gold_pu, gold_rows_fine):
    MAP,ORDER=GRAN[gran]
    stage2_valid = False; key_pu = True
    d=pd.read_csv(path,dtype=str,keep_default_na=False)
    if key_pu:
        gold={k:MAP[v] for k,v in gold_pu.items()}
        Nc=collections.Counter(gold.values())
        pred=collections.defaultdict(lambda:{"valid":[],"fp":[]})
        for _,r in d.iterrows():
            pl=N2ID.get(r["pred_label_name"].strip())
            if pl is None: continue
            pl=MAP[pl]; mr=r["match_rule"]; pid=E._normalize_paper_id(r["paper_id"])
            if mr=="genuine_fp":
                for u in E._extract_norm_urls_from_field(r["url_output"] or r["url_output_norm"]): pred[(pid,u)]["fp"].append(pl)
            elif mr=="stage2" and not stage2_valid:
                for u in E._extract_norm_urls_from_field(r["url_gold"] or r["url_gold_norm"]): pred[(pid,u)]["fp"].append(pl)
            else:
                for u in E._extract_norm_urls_from_field(r["url_gold"] or r["url_gold_norm"]): pred[(pid,u)]["valid"].append(pl)
        TP=collections.Counter(); FP=collections.Counter()
        for k in set(gold)|set(pred):
            gc=gold.get(k); vlab=pred[k]["valid"]; flab=pred[k]["fp"]
            if gc is not None and vlab and majority(vlab)==gc: TP[gc]+=1
            else:
                wrong=vlab+flab
                if wrong: FP[majority(wrong)]+=1
        FN={g:Nc[g]-TP[g] for g in ORDER}
        return TP,FP,FN,Nc,ORDER
    else:
        Nc=collections.Counter(MAP[c] for c in gold_rows_fine)
        TP=collections.Counter(); FP=collections.Counter()
        for _,r in d.iterrows():
            pl=N2ID.get(r["pred_label_name"].strip())
            if pl is None: continue
            pl=MAP[pl]; mr=r["match_rule"]
            if mr=="genuine_fp" or (mr=="stage2" and not stage2_valid): FP[pl]+=1
            else:
                gc=N2ID.get(r["gold_label_name"].strip())
                gc=MAP[gc] if gc is not None else None
                if gc==pl: TP[gc]+=1
                else: FP[pl]+=1
        FN={g:Nc[g]-TP[g] for g in ORDER}
        return TP,FP,FN,Nc,ORDER

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold",default=".")
    ap.add_argument("--pred_dir",nargs="+",default=["."])
    ap.add_argument("--granularity",choices=["fine","coarse","binary"],default="fine")
    a=ap.parse_args()
    gold_pu,gold_rows=load_gold(a.gold); pipes=discover(a.pred_dir)
    for name,path in pipes.items():
        TP,FP,FN,Nc,ORDER=score(path,a.granularity,gold_pu,gold_rows)
        print(f"\n  {name}")
        print(f"    {'class':<26}{'P':>7}{'R':>7}{'F1':>7}")
        Fs=[]; tT=tF=tN=0
        for g in ORDER:
            P,R,Fl=prf(TP[g],FP[g],FN[g]); Fs.append(Fl); tT+=TP[g];tF+=FP[g];tN+=FN[g]
            print(f"    {g:<26}{P:>7.3f}{R:>7.3f}{Fl:>7.3f}")
        miP,miR,miF=prf(tT,tF,tN)
        print(f"    {'MACRO':<26}{'':7}{'':7}{sum(Fs)/len(Fs):>7.3f}")
        print(f"    {'MICRO':<26}{miP:>7.3f}{miR:>7.3f}{miF:>7.3f}")

if __name__=="__main__": main()
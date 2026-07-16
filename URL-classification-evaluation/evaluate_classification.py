#!/usr/bin/env python3
"""
End-to-end (extraction -> EnSU classification) scoring.
Per-class P / R / F1 + macro + micro, for every pipeline.

Match unit = (paper_id, url).
  recovery match rules: stage0/stage3 (exact URL), stage2 (near URL), fallback.
  genuine_fp = corrupted string whose twin gold URL and label are known.
  TP = gold unit recovered by a VALID pass AND labeled correctly.
  FN = N - TP.  FP = genuine_fp (always) + stage2 (variants 1-2) + mislabeled valid.

VARIANTS
  1: stage2 -> FP ;    key distinct (paper_id,url)     2: stage2 -> FP ;    key gold rows
  3: stage2 -> valid ; key distinct (paper_id,url)     4: stage2 -> valid ; key gold rows

GRANULARITY (label mapping applied to BOTH gold and predicted labels, then scored):
  fine   : 6 classes (general-url, third/author dataset, third/author software, project)
  coarse : 4 groups  (general-url; dataset=1+2; software=3+4; project)   [Option A]
  binary : 2 groups  (OADS = classes 1-5; not-OADS = general-url)

Usage:
  python3 end_to_end_variants.py --gold GOLD_DIR --pred_dir PRED_DIR
      [--variant N] [--granularity fine|coarse|binary] [--out perclass.csv]
Pipelines auto-discovered from PRED_DIR/*_all_predictions.csv (excludes *_gold_*).
Requires exact_url_evaluator.py importable.
"""
import os, glob, argparse, collections
import pandas as pd
import exact_url_evaluator as E

FINE={0:"general-url",1:"third-party-dataset",2:"author-provided-dataset",
      3:"third-party-software",4:"author-provided-software",5:"project"}
N2ID={v:k for k,v in FINE.items()}
# label-id -> group-name, and display order, per granularity
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
    for f in sorted(glob.glob(os.path.join(gold_dir,"*-gold.csv"))):
        d=pd.read_csv(f,dtype=str,keep_default_na=False)
        for _,r in d.iterrows():
            if str(r.get("split","")).strip()!="test" or str(r.get("Label","")).strip()=="": continue
            c=int(float(r["Label"].split('.')[0])); pid=E._normalize_paper_id(r["paper_id"])
            rows.append(c)
            for u in E._extract_norm_urls_from_field(r["url"]): pu[(pid,u)][c]+=1
    gold_pu={k:v.most_common(1)[0][0] for k,v in pu.items()}    # fine class ids
    return gold_pu, rows                                        # rows = list of fine class ids

def discover(pred_dirs):
    out={}
    for pd_ in pred_dirs:
        for f in sorted(glob.glob(os.path.join(pd_,"*_all_predictions.csv"))):
            b=os.path.basename(f)
            if b.endswith("_gold_all_predictions.csv"): continue
            out[b[:-len("_all_predictions.csv")]]=f
    return out

def score(path, variant, gran, gold_pu, gold_rows_fine):
    MAP,ORDER=GRAN[gran]
    stage2_valid = variant in (3,4); key_pu = variant in (1,3)
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

DESC={1:"stage2->FP, key (paper,url)",2:"stage2->FP, key rows",
      3:"stage2 valid, key (paper,url)",4:"stage2 valid, key rows"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold",required=True)
    ap.add_argument("--pred_dir",required=True,nargs="+",help="one or more folders containing *_all_predictions.csv")
    ap.add_argument("--granularity",choices=["fine","coarse","binary"],default="fine")
    ap.add_argument("--out")
    a=ap.parse_args()
    gold_pu,gold_rows=load_gold(a.gold); pipes=discover(a.pred_dir)
    print(f"variant {1} ({DESC[1]}) | granularity {a.granularity} | pipelines: {list(pipes)}")
    csv=[]
    for name,path in pipes.items():
        TP,FP,FN,Nc,ORDER=score(path,1,a.granularity,gold_pu,gold_rows)
        print(f"\n  {name}")
        print(f"    {'class':<26}{'P':>7}{'R':>7}{'F1':>7}")
        Fs=[]; tT=tF=tN=0
        for g in ORDER:
            P,R,Fl=prf(TP[g],FP[g],FN[g]); Fs.append(Fl); tT+=TP[g];tF+=FP[g];tN+=FN[g]
            print(f"    {g:<26}{P:>7.3f}{R:>7.3f}{Fl:>7.3f}")
            csv.append([1,a.granularity,name,g,TP[g],FP[g],FN[g],round(P,4),round(R,4),round(Fl,4)])
        miP,miR,miF=prf(tT,tF,tN)
        print(f"    {'MACRO':<26}{'':7}{'':7}{sum(Fs)/len(Fs):>7.3f}")
        print(f"    {'MICRO':<26}{miP:>7.3f}{miR:>7.3f}{miF:>7.3f}")
        csv.append([1,a.granularity,name,"MACRO","","","","","",round(sum(Fs)/len(Fs),4)])
        csv.append([1,a.granularity,name,"MICRO",tT,tF,tN,round(miP,4),round(miR,4),round(miF,4)])
    if a.out:
        pd.DataFrame(csv,columns=["variant","granularity","pipeline","class","TP","FP","FN","P","R","F1"]).to_csv(a.out,index=False)
        print(f"\n-> {a.out}")

if __name__=="__main__": main()
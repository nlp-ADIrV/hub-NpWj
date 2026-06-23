"""
第八周文本匹配作业
"""
import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# ===================== 全局固定配置 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
random.seed(42)
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
OUT = BASE / "outputs"
CKPT = OUT / "checkpoints"
FIG = OUT / "figures"
LOG = OUT / "logs"
for p in [DATA_DIR, CKPT, FIG, LOG]:
    p.mkdir(exist_ok=True, parents=True)
BERT_PATH = BASE / "pretrain_models/bert-base-chinese"
EPOCHS = 3
BATCH = 32
LR = 2e-5
WARMUP = 0.1
SINGLE_MAX = 64
PAIR_MAX = 128
MARGIN = 0.3
# ===================== 1. 数据集下载工具 =====================
def save_jsonl(save_path, rows):
    with open(save_path, "w", encoding="utf-8") as fp:
        for r in rows:
            json.dump(r, fp, ensure_ascii=False)
            fp.write("\n")

def norm_row(raw):
    s1 = raw.get("sentence1", raw.get("text1", ""))
    s2 = raw.get("sentence2", raw.get("text2", ""))
    lab = int(raw.get("label", raw.get("score", 0)))
    return {"sentence1": str(s1), "sentence2": str(s2), "label": lab}

def stat_rows(rows, name):
    total = len(rows)
    pos = sum(i["label"] == 1 for i in rows)
    neg = total - pos
    print(f"{name} | 总{total} 相似{pos}({pos/total*100:.1f}%) 不相似{neg}")

def preview(path, n=3):
    print(f"\n样本预览 {path}:")
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= n: break
            d = json.loads(line)
            tag = "相似" if d["label"] else "不相似"
            print(f"[{tag}] {d['sentence1']} | {d['sentence2']}")

def download_all_datasets():
    print("===== 开始下载三套数据集 ====")
    # AFQMC
    print("\n1. AFQMC(蚂蚁金融)")
    ds_af = load_dataset("clue", "afqmc")
    af_dir = DATA_DIR / "afqmc"
    af_dir.mkdir(exist_ok=True, parents=True)
    for split in ["train", "validation", "test"]:
        data = [norm_row(i) for i in ds_af[split]]
        save_jsonl(af_dir / f"{split}.jsonl", data)
        stat_rows(data, f"AFQMC-{split}")
    preview(af_dir / "train.jsonl")
    # LCQMC
    print("\n2. LCQMC(通用口语)")
    ds_lc = load_dataset("C-MTEB/LCQMC")
    lc_dir = DATA_DIR / "lcqmc"
    lc_dir.mkdir(exist_ok=True, parents=True)
    for split in ["train", "validation", "test"]:
        data = [norm_row(i) for i in ds_lc[split]]
        save_jsonl(lc_dir / f"{split}.jsonl", data)
        stat_rows(data, f"LCQMC-{split}")
    preview(lc_dir / "train.jsonl")
    # BQ Corpus
    print("\n3. BQ Corpus(银行)")
    ds_bq = load_dataset("FinanceMTEB/bq_corpus", split="test")
    all_bq = [norm_row(i) for i in ds_bq]
    random.shuffle(all_bq)
    n = len(all_bq)
    val = all_bq[:int(n*0.1)]
    test = all_bq[int(n*0.1):int(n*0.2)]
    train = all_bq[int(n*0.2):]
    bq_dir = DATA_DIR / "bq_corpus"
    bq_dir.mkdir(exist_ok=True, parents=True)
    save_jsonl(bq_dir / "train.jsonl", train)
    save_jsonl(bq_dir / "validation.jsonl", val)
    save_jsonl(bq_dir / "test.jsonl", test)
    stat_rows(train, "BQ-train")
    preview(bq_dir / "train.jsonl")
    print("\n全部数据集下载完成，路径：", DATA_DIR)

def load_json(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            res.append(json.loads(line.strip()))
    return res
# ===================== 2. 数据集类 =====================
tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
class PairData(Dataset):
    def __init__(self, file):
        self.data = load_json(file)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        s1, s2, lab = d["sentence1"], d["sentence2"], d["label"]
        e1 = tokenizer(s1, max_length=SINGLE_MAX, truncation=True, padding="max_length", return_tensors="pt")
        e2 = tokenizer(s2, max_length=SINGLE_MAX, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "ia": e1["input_ids"].squeeze(0), "aa": e1["attention_mask"].squeeze(0), "ta": e1["token_type_ids"].squeeze(0),
            "ib": e2["input_ids"].squeeze(0), "ab": e2["attention_mask"].squeeze(0), "tb": e2["token_type_ids"].squeeze(0),
            "label": torch.tensor(lab)
        }

class TripData(Dataset):
    def __init__(self, file):
        raw = load_json(file)
        pos = [i for i in raw if i["label"] == 1]
        texts = list(set(i["sentence1"]+i["sentence2"] for i in raw))
        self.trip = []
        for p in pos:
            anc = p["sentence1"]
            pos_txt = p["sentence2"]
            neg = random.choice(texts)
            while neg == anc or neg == pos_txt:
                neg = random.choice(texts)
            self.trip.append((anc, pos_txt, neg))
    def __len__(self): return len(self.trip)
    def __getitem__(self, idx):
        a,p,n = self.trip[idx]
        ea = tokenizer(a, max_length=SINGLE_MAX, padding="max_length", truncation=True, return_tensors="pt")
        ep = tokenizer(p, max_length=SINGLE_MAX, padding="max_length", truncation=True, return_tensors="pt")
        en = tokenizer(n, max_length=SINGLE_MAX, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "ia":ea["input_ids"].squeeze(0),"aa":ea["attention_mask"].squeeze(0),"ta":ea["token_type_ids"].squeeze(0),
            "ip":ep["input_ids"].squeeze(0),"ap":ep["attention_mask"].squeeze(0),"tp":ep["token_type_ids"].squeeze(0),
            "in":en["input_ids"].squeeze(0),"an":en["attention_mask"].squeeze(0),"tn":en["token_type_ids"].squeeze(0),
        }

class CrossData(Dataset):
    def __init__(self, file):
        self.data = load_json(file)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        s1,s2,lab = d["sentence1"],d["sentence2"],d["label"]
        enc = tokenizer(s1,s2,max_length=PAIR_MAX,truncation=True,padding="max_length",return_tensors="pt")
        return {
            "input_ids":enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "token_type_ids":enc["token_type_ids"].squeeze(0),
            "label":torch.tensor(lab)
        }
# ===================== 3. 模型定义 =====================
class BiEncoder(nn.Module):
    def __init__(self, pool="mean", layer=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(BERT_PATH))
        self.bert.encoder.layer = self.bert.encoder.layer[:layer]
        self.pool = pool
    def pool_out(self, hidden, mask):
        if self.pool == "cls":
            return hidden[:,0,:]
        mask = mask.unsqueeze(-1)
        if self.pool == "mean":
            sum_h = (hidden * mask).sum(1)
            cnt = mask.sum(1).clamp(1e-8)
            return sum_h / cnt
        if self.pool == "max":
            fill = hidden + (1-mask)*-1e9
            return fill.max(1).values
    def forward(self, batch_a, batch_b):
        oa = self.bert(**batch_a)
        ob = self.bert(**batch_b)
        va = self.pool_out(oa.last_hidden_state, batch_a["attention_mask"])
        vb = self.pool_out(ob.last_hidden_state, batch_b["attention_mask"])
        va = F.normalize(va, p=2, dim=-1)
        vb = F.normalize(vb, p=2, dim=-1)
        sim = F.cosine_similarity(va, vb)
        return va, vb, sim

class CrossEncoder(nn.Module):
    def __init__(self, layer=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(BERT_PATH))
        self.bert.encoder.layer = self.bert.encoder.layer[:layer]
        self.head = nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, batch):
        out = self.bert(**batch)
        cls = out.last_hidden_state[:,0,:]
        return self.head(cls)
# ===================== 4. 评估&绘图 =====================
def eval_bi(model, loader, data_name):
    model.eval()
    sims_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            ba = {"input_ids":batch["ia"].cuda(),"attention_mask":batch["aa"].cuda(),"token_type_ids":batch["ta"].cuda()}
            bb = {"input_ids":batch["ib"].cuda(),"attention_mask":batch["ab"].cuda(),"token_type_ids":batch["tb"].cuda()}
            _, _, sim = model(ba, bb)
            sims_all.extend(sim.cpu().tolist())
            labels_all.extend(batch["label"].tolist())
    best_f1, best_t, best_acc = 0, 0.5, 0
    for t in np.arange(0,1,0.01):
        pred = [1 if s>t else 0 for s in sims_all]
        f = f1_score(labels_all, pred)
        if f>best_f1:
            best_f1 = f
            best_t = t
            best_acc = accuracy_score(labels_all, pred)
    # 绘图
    pos = [s for s,l in zip(sims_all,labels_all) if l==1]
    neg = [s for s,l in zip(sims_all,labels_all) if l==0]
    plt.figure(figsize=(8,4))
    plt.hist(pos, alpha=0.6, label="相似样本")
    plt.hist(neg, alpha=0.6, label="不相似样本")
    plt.axvline(x=best_t, ls="--", c="k", label=f"阈值{best_t:.2f}")
    plt.legend()
    plt.title(f"{data_name} 相似度分布")
    plt.savefig(FIG/f"{data_name}_sim.png", dpi=150)
    plt.close()
    return best_acc, best_f1, best_t

def eval_cross(model, loader):
    model.eval()
    preds = []
    labs = []
    with torch.no_grad():
        for batch in loader:
            b = {k:batch[k].cuda() for k in batch if k!="label"}
            logits = model(b)
            p = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(p)
            labs.extend(batch["label"].tolist())
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average="binary")
    return acc, f1

# ===================== 5. 训练函数 =====================
def train_bi(data_dir, loss_type, pool):
    train_path = DATA_DIR / data_dir / "train.jsonl"
    val_path = DATA_DIR / data_dir / "validation.jsonl"
    if loss_type == "cos":
        train_ds = PairData(train_path)
    else:
        train_ds = TripData(train_path)
    val_ds = PairData(val_path)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    model = BiEncoder(pool=pool).cuda()
    opt = AdamW(model.parameters(), lr=LR)
    total_step = len(train_loader)*EPOCHS
    warm = int(total_step*WARMUP)
    scheduler = get_linear_schedule_with_warmup(opt, warm, total_step)
    best_f1 = 0
    log_list = []
    for e in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        bar = tqdm(train_loader, desc=f"{data_dir} E{e}")
        for batch in bar:
            ba = {"ia":batch["ia"].cuda(),"aa":batch["aa"].cuda(),"ta":batch["ta"].cuda()}
            bb = {"ib":batch["ib"].cuda(),"ab":batch["ab"].cuda(),"tb":batch["tb"].cuda()}
            if loss_type == "cos":
                va, vb, sim = model(ba, bb)
                target = torch.where(batch["label"].cuda()==1, torch.tensor(1.).cuda(), torch.tensor(-1.).cuda())
                loss = F.cosine_embedding_loss(va, vb, target, margin=MARGIN)
            else:
                ba_trip = {"ia":batch["ia"].cuda(),"aa":batch["aa"].cuda(),"ta":batch["ta"].cuda()}
                bp = {"ip":batch["ip"].cuda(),"ap":batch["ap"].cuda(),"tp":batch["tp"].cuda()}
                bn = {"in":batch["in"].cuda(),"an":batch["an"].cuda(),"tn":batch["tn"].cuda()}
                va, _, _ = model(ba_trip, ba_trip)
                vp, _, _ = model(bp, bp)
                vn, _, _ = model(bn, bn)
                loss = F.triplet_margin_loss(va, vp, vn, margin=MARGIN)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()
            total_loss += loss.item()
            bar.set_postfix(loss=round(loss.item(),4))
        avg_loss = total_loss / len(train_loader)
        acc, f1, thr = eval_bi(model, val_loader, data_dir)
        print(f"Epoch{e} loss:{avg_loss:.4f} Acc:{acc:.4f} F1:{f1:.4f} thr:{thr}")
        log_list.append({"epoch":e,"loss":avg_loss,"acc":acc,"f1":f1,"thr":thr})
        if f1>best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), CKPT/f"bi_{data_dir}_{loss_type}_{pool}.pt")
    with open(LOG/f"bi_{data_dir}_{loss_type}_log.json","w",encoding="utf-8") as f:
        json.dump(log_list, f)

def train_cross(data_dir):
    train_p = DATA_DIR/data_dir/"train.jsonl"
    val_p = DATA_DIR/data_dir/"validation.jsonl"
    train_ds = CrossData(train_p)
    val_ds = CrossData(val_p)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    model = CrossEncoder().cuda()
    loss_fn = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=LR)
    total_step = len(train_loader)*EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, int(total_step*WARMUP), total_step)
    best_f1 = 0
    for e in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        bar = tqdm(train_loader)
        for batch in bar:
            b = {k:batch[k].cuda() for k in batch if k!="label"}
            lab = batch["label"].cuda()
            logits = model(b)
            loss = loss_fn(logits, lab)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()
            total_loss += loss.item()
            bar.set_postfix(loss=round(loss.item(),4))
        avg_loss = total_loss/len(train_loader)
        acc, f1 = eval_cross(model, val_loader)
        print(f"Epoch{e} loss:{avg_loss:.4f} Acc:{acc:.4f} F1:{f1:.4f}")
        if f1>best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), CKPT/f"cross_{data_dir}.pt")
# ===================== 6. 传统基线 =====================
def jaccard(s1, s2, gram=1):
    set1 = set([s1[i:i+gram] for i in range(len(s1)-gram+1)])
    set2 = set([s2[i:i+gram] for i in range(len(s2))])
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter/union if union>0 else 0

def run_baseline(data_name):
    path = DATA_DIR/data_name
    train = load_json(path/"train.jsonl")
    val = load_json(path/"validation.jsonl")
    tr_s1,tr_s2,tr_y = zip(*[(d["sentence1"],d["sentence2"],d["label"]) for d in train])
    va_s1,va_s2,va_y = zip(*[(d["sentence1"],d["sentence2"],d["label"]) for d in val])
    # 单字符Jaccard
    j_pred = [1 if jaccard(s1,s2)>0.3 else 0 for s1,s2 in zip(va_s1,va_s2)]
    j_acc = accuracy_score(va_y, j_pred)
    j_f1 = f1_score(va_y, j_pred)
    # 二元Jaccard
    bi_pred = [1 if jaccard(s1,s2,2)>0.25 else 0 for s1,s2 in zip(va_s1,va_s2)]
    bi_acc = accuracy_score(va_y, bi_pred)
    bi_f1 = f1_score(va_y, bi_pred)
    # TFIDF+SVM
    tf = TfidfVectorizer(ngram_range=(1,2))
    tf.fit(tr_s1+tr_s2)
    tr_feat = np.hstack([tf.transform(tr_s1).toarray(), tf.transform(tr_s2).toarray()])
    va_feat = np.hstack([tf.transform(va_s1).toarray(), tf.transform(va_s2).toarray()])
    svm = LinearSVC()
    svm.fit(tr_feat, tr_y)
    svm_pred = svm.predict(va_feat)
    svm_acc = accuracy_score(va_y, svm_pred)
    svm_f1 = f1_score(va_y, svm_pred)
    print(f"\n【{data_name} 基线指标】")
    print(f"字符Jaccard acc={j_acc:.3f} f1={j_f1:.3f}")
    print(f"二元Jaccard acc={bi_acc:.3f} f1={bi_f1:.3f}")
    print(f"TF-IDF+SVM acc={svm_acc:.3f} f1={svm_f1:.3f}")
# ===================== 7. 错误样本分析 =====================
def bad_case_analysis(data_name, ckpt_name):
    val_path = DATA_DIR/data_name/"validation.jsonl"
    val_ds = PairData(val_path)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    model = BiEncoder().cuda()
    model.load_state_dict(torch.load(CKPT/ckpt_name))
    acc, f1, thr = eval_bi(model, val_loader, data_name)
    raw = load_json(val_path)
    fp, fn = [], []
    model.eval()
    idx = 0
    with torch.no_grad():
        for batch in val_loader:
            ba = {"ia":batch["ia"].cuda(),"aa":batch["aa"].cuda(),"ta":batch["ta"].cuda()}
            bb = {"ib":batch["ib"].cuda(),"ab":batch["ab"].cuda(),"tb":batch["tb"].cuda()}
            _, _, sim = model(ba, bb)
            sims = sim.cpu().tolist()
            labs = batch["label"].tolist()
            for s,l in zip(sims, labs):
                pred = 1 if s>thr else 0
                item = raw[idx]
                idx += 1
                if pred ==1 and l ==0:
                    fp.append((s, item["sentence1"], item["sentence2"]))
                if pred ==0 and l ==1:
                    fn.append((s, item["sentence1"], item["sentence2"]))
    print(f"\n{data_name} 错误样本统计 FP:{len(fp)} FN:{len(fn)}")
    print("FP(预测相似实际不相似)前5条：")
    for score,s1,s2 in fp[:5]:
        print(f"{score:.2f} | {s1} {s2}")
    print("FN(预测不相似实际相似)前5条：")
    for score,s1,s2 in fn[:5]:
        print(f"{score:.2f} | {s1} {s2}")
# ===================== 入口主函数 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第八周文本匹配单文件作业")
    parser.add_argument("--mode", required=True, choices=["download","train_bi","train_cross","baseline","badcase"])
    parser.add_argument("--dataset", default="lcqmc", choices=["afqmc","lcqmc","bq_corpus"])
    parser.add_argument("--loss", default="cos", choices=["cos","trip"])
    parser.add_argument("--pool", default="mean", choices=["cls","mean","max"])
    parser.add_argument("--ckpt", default="bi_lcqmc_cos_mean.pt")
    args = parser.parse()
    if args.mode == "download":
        download_all_datasets()
    elif args.mode == "train_bi":
        train_bi(args.dataset, args.loss, args.pool)
    elif args.mode == "train_cross":
        train_cross(args.dataset)
    elif args.mode == "baseline":
        run_baseline(args.dataset)
    elif args.mode == "badcase":
        bad_case_analysis(args.dataset, args.ckpt)

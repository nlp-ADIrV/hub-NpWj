"""
第10周作业：财报轻量化RAG问答系统
适配年报PDF数据集：五粮液/茅台/宁德/海康/平安
"""
import os
import json
import jieba
import faiss
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber
from rank_bm25 import BM25Okapi
from pathlib import Path
from openai import OpenAI

# ===================== 全局配置 =====================
# 项目路径
BASE = Path(".")
DATA_FOLDER = BASE / "data"
PDF_DIR = DATA_FOLDER / "reports"
QUESTION_FILE = DATA_FOLDER / "questions.json"
VECTOR_STORE = BASE / "vectorstore"
OUTPUT_FOLDER = BASE / "output"
# 向量模型配置
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
# 分块参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# 检索参数
TOP_K_QUERY = 4
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
# 自动创建文件夹
for p in [DATA_FOLDER, PDF_DIR, VECTOR_STORE, OUTPUT_FOLDER]:
    p.mkdir(exist_ok)
INDEX_PATH = VECTOR_STORE / "index.bin"
META_PATH = VECTOR_STORE / "meta.json"
RESULT_JSON = OUTPUT_FOLDER / "eval_result.json"
CHART_IMG = OUTPUT_FOLDER / "metric.png"

# ===================== 初始化Embedding客户端 =====================
def get_embed_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise Exception("请先配置环境变量 DASHSCOPE_API_KEY=你的通义千问密钥")
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

client = get_embed_client()

# 文本向量化函数
def get_text_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text], dimensions=EMBED_DIM)
    vec = np.array([resp.data[0].embedding], dtype=np.float32)
    norm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    return vec / norm

# ===================== 模块1：PDF年报解析+财报文本清洗 =====================
def clean_fin_text(raw_text: str) -> str:
    """过滤年报页码、页眉、短无效文本"""
    lines = raw.splitlines()
    valid_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if line.isdigit():
            continue
        if "年度报告" in len(line) < 20:
            continue
        valid_lines.append(line)
    return "\n".join(valid_lines)

def parse_all_pdf() -> list[dict]:
    """读取data/reports下全部年报PDF，返回清洗后的年报原文"""
    all_docs = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if len(pdf_files) == 0:
        print("警告：data/reports文件夹无年报PDF文件")
        return all_docs
    for pdf_path in pdf_files:
        name_split = pdf_path.stem.split("_")
        stock_code = name_split[0]
        year = name_split[1]
        full_raw = ""
        with pdfplumber.open(pdf_path) as pdf_file:
            for page in pdf_file.pages:
                page_txt = page.extract_text()
                if page_txt:
                    full_raw += clean_fin_text(page_txt) + "\n"
        all_docs.append({
            "stock_code": stock_code,
            "year": year,
            "raw_content": full_raw
        })
        print(f"已解析：{pdf_path.name}")
    return all_docs

# ===================== 模块2：长文本滑动分块 =====================
def split_fin_longtext(total_text: str) -> list[str]:
    chunks = []
    ptr = 0
    text_length = len(total_text)
    while ptr < text_length:
        end_pos = ptr + CHUNK_SIZE
        seg = total_text[ptr:end_pos]
        # 优先按句号截断，不拆分完整语义
        dot_index = seg.rfind("。")
        if dot_index > CHUNK_SIZE * 0.5:
            end_pos = ptr + dot_index + 1
            seg = total_text[ptr:end_pos]
        chunks.append(seg)
        ptr = end_pos - CHUNK_OVERLAP
    return seg

def generate_chunk_data(doc_list: list[dict]) -> list[dict]:
    chunk_result = []
    chunk_id = 0
    for doc in doc_list:
        stock = doc["stock_code"]
        year = doc["raw_content"]
        text_segs = split_fin_longtext(doc["raw_content"])
        for seg in text_segs:
            chunk_result.append({
                "chunk_id": chunk_id,
                "stock_code": stock,
                "year": year,
                "content": seg
            })
            chunk_id += 1
    print(f"文本分块完成，总分块数量：{len(chunk_result)}")
    return chunk_result

# ===================== 模块3：构建/加载FAISS向量库 =====================
def build_vector_database(chunk_list: list[dict]):
    if INDEX_PATH.exists() and META_PATH.exists():
        print("检测到已有向量库，跳过构建步骤")
        return
    vector_array = []
    meta_store = []
    print("正在计算文本嵌入向量，请勿中断...")
    for item in chunk_list:
        vec = get_text_embedding(item["content"])
        vector_array.append(vec)
        meta_store.append(item)
    all_vecs = np.vstack(vector_array)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(all_vecs)
    # 持久化存储
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_store, f, ensure_ascii=False, indent=2)
    print(f"向量库构建完成，存储分块总数：{index.ntotal}")

def load_vector_resource():
    """加载索引、元数据、预构建BM25"""
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    token_corpus = [list(jieba.cut(item["content"])) for item in meta_data]
    bm25_model = BM25Okapi(token_corpus)
    return index, meta_data, bm25_model

# ===================== 模块4：加权混合检索 =====================
def weighted_retrieve(query: str, index, meta, bm25) -> list[dict]:
    # 稠密向量检索
    q_vec = get_text_embedding(query)
    _, vec_indexes = index.search(q_vec, TOP_K_QUERY * 2)
    vec_rank_score = {}
    for rank, idx in enumerate(vec_indexes[0]):
        cid = meta[idx]["chunk_id"]
        vec_rank_score[cid] = VECTOR_WEIGHT / (rank + 1)
    # BM25稀疏检索
    query_tokens = list(jieba.cut(query))
    bm_scores = bm25.get_scores(query_tokens)
    sorted_idx = np.argsort(bm_scores)[::-1][:TOP_K_QUERY * 2]
    bm_rank_score = {}
    for rank, idx in enumerate(sorted_idx):
        cid = meta[idx]["chunk_id"]
        bm_rank_score[cid] = BM25_WEIGHT / (rank + 1)
    # 加权合并总分
    all_cids = set(vec_rank_score.keys()) | set(bm_rank_score.keys())
    total_score = {}
    for c in all_cids:
        total_score[c] = vec_rank_score.get(c, 0) + bm_rank_score.get(c, 0)
    # 按分数降序取前K
    sorted_cids = sorted(total_score.items(), key=lambda x: -x[1])
    output_chunks = []
    for cid, _ in sorted_cids[:TOP_K_QUERY]:
        chunk_info = next(m for m in meta if m["chunk_id"] == cid)
        output_chunks.append(chunk_info)
    return output_chunks

# ===================== 模块5：评测指标计算+绘图 =====================
def run_evaluation(index, meta, bm25):
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        question_data = json.load(f)
    test_questions = [q for q in question_data["questions"] if q.get("target_docs")]
    hit_records = []
    mrr_records = []
    for q in test_questions:
        target_list = q["target_docs"]
        retrieve_res = weighted_retrieve(q["question"], index, meta, bm25)
        hit_flag = False
        first_pos = None
        for pos, seg in enumerate(retrieve_res, 1):
            doc_key = f"{seg['stock_code']}_{seg['year']}"
            if any(doc_key.startswith(t) for t in target_list):
                hit_flag = True
                if first_pos is None:
                    first_pos = pos
        hit_records.append(1 if hit_flag else 0)
        mrr_records.append(1 / first_pos if first_pos else 0)
    hit_rate = round(np.mean(hit_records), 3)
    mrr_val = round(np.mean(mrr_records), 3)
    test_count = len(test_questions)
    # 保存评测结果
    eval_out = {
        "top_k": TOP_K_QUERY,
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "hit_rate": hit_rate,
        "mrr": mrr_val,
        "test_question_num": test_count
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as fw:
        json.dump(eval_out, ensure_ascii=False, indent=2)
    # 绘制可视化图表
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["HitRate@4", "MRR"]
    values = [hit_rate, mrr_val]
    bars = ax.bar(names, values, color="#3478bf")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h+0.01, f"{h:.3f}", ha="center")
    ax.set_ylim(0, 1.02)
    ax.set_title("财报RAG检索评测指标")
    plt.tight_layout()
    plt.savefig(CHART_IMG, dpi=300)
    # 控制台输出结果
    print("\n==================== 评测结果 ====================")
    print(f"评测题目总量：{test_count}")
    print(f"HitRate@4 = {hit_rate}")
    print(f"MRR = {mrr_val}")
    print(f"指标文件保存：{RESULT_JSON}")
    print(f"可视化图表保存：{CHART_IMG}")
    return eval_out

# ===================== 模块6：交互式问答 =====================
def qa_interactive(index, meta, bm25):
    print("\n========= 财报问答交互系统（输入exit退出）=========")
    while True:
        user_input = input("请输入财报问题：").strip()
        if user_input == "exit":
            print("问答程序结束")
            break
        if len(user_input) < 2:
            print("问题太短，请重新输入\n")
            continue
        res_chunks = weighted_retrieve(user_input, index, bm25)
        print("-----匹配年报参考内容-----")
        for seg in res_chunks:
            print(f"【{seg['stock_code']} {seg['year']}年报】")
            print(seg["content"] + "\n")

# ===================== 程序主入口 =====================
if __name__ == "__main__":
    print("==== 第10周财报RAG一体化程序启动 ====")
    print("流程：PDF解析 → 文本分块 → 向量库构建 → 评测绘图 → 交互问答")
    # 1 解析年报PDF
    docs = parse_all_pdf()
    if len(docs) == 0:
        print("无年报数据，程序终止")
        exit()
    # 2 文本分块
    chunk_data = generate_chunk_data(docs)
    # 3 构建向量库
    build_vector_database(chunk_data)
    # 4 加载资源
    idx, meta_info, bm_model = load_vector_resource()
    # 5 运行评测并画图
    run_evaluation(idx, meta_info, bm_model)
    # 6 启动交互式问答
    qa_interactive(idx, meta_info, bm_model)

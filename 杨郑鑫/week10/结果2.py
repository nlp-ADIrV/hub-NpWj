(py312) PS D:\BaiduNetdiskDownload\week10检索增强生成\week10 检索增强生成RAG\week10 检索增强生成RAG\rag_annual_report> python src_langchain/rag_chain_lc.py --query "成市监规〔2023〕6 号这个是什么政策"
2026-07-09 20:53:02,955 [INFO] 加载 embedding 模型...
2026-07-09 20:53:10,632 [INFO] Loading SentenceTransformer model from D:\BaiduNetdiskDownload\week10检索增强生成\week10 检索增强生成RAG\week10 检索增强生成RAG\rag_annual_report\models\bge-small-zh-v1.5.
2026-07-09 20:53:10,707 [INFO] 加载向量库...
2026-07-09 20:53:10,723 [INFO] Loading faiss with AVX2 support.
2026-07-09 20:53:10,723 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-09 20:53:10,723 [INFO] Loading faiss.
2026-07-09 20:53:10,819 [INFO] Successfully loaded faiss.

============================================================
问题：成市监规〔2023〕6 号这个是什么政策
============================================================
2026-07-09 20:53:14,970 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

成市监规〔2023〕6 号是《成都市农民专业合作社登记（备案）自然人身份信息管理工作规范》的文号，由**成都市市场监督管理局、成都市农业农村局**联合印发（来源：[2] 第1页）。

该政策旨在“持续优化营商环境，解决农民专业合作社成员众多、身份信息实名认证难等问题，切实为农民专业合作社登记（备案）提供便利”，依据《中华人民共和国农民专业合作社法》《中华人民共和国市场主体登记管理条例 》及其实施细则等法律法规制定（来源：[2] 第1页）。

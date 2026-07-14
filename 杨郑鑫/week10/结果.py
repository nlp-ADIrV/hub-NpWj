"""
(py312) PS D:\BaiduNetdiskDownload\week10检索增强生成\week10 检索增强生成RAG\week10 检索增强生成RAG\rag_annual_report> python src/rag_pipeline.py --query "成市监规〔2023〕6 号这个是什么政策" --no-rerank
2026-07-08 22:32:24,464 [INFO] Loading faiss with AVX2 support.
2026-07-08 22:32:24,464 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 22:32:24,464 [INFO] Loading faiss.
2026-07-08 22:32:24,487 [INFO] Successfully loaded faiss.
2026-07-08 22:32:24,499 [INFO] FAISS 索引加载完成，共 98 条向量
D:\Anaconda_envs\envs\py312\Lib\site-packages\jieba\_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
2026-07-08 22:32:24,603 [INFO] 构建 BM25 索引（分词中，请稍候）...
Building prefix dict from the default dictionary ...
2026-07-08 22:32:24,604 [DEBUG] Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
2026-07-08 22:32:24,604 [DEBUG] Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
Loading model cost 0.608 seconds.
2026-07-08 22:32:25,211 [DEBUG] Loading model cost 0.608 seconds.
Prefix dict has been built successfully.
2026-07-08 22:32:25,212 [DEBUG] Prefix dict has been built successfully.
2026-07-08 22:32:25,321 [INFO] BM25 索引完成
2026-07-08 22:32:26,269 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 22:32:26,273 [INFO] 向量召回: 10 条，最高分=0.721
2026-07-08 22:32:26,274 [INFO] BM25 召回: 10 条，RRF 后: 17 条
2026-07-08 22:32:26,275 [INFO] 最终使用 4 条上下文，最高向量相似度=0.721
2026-07-08 22:32:30,198 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：成市监规〔2023〕6 号这个是什么政策
============================================================

根据提供的参考资料，**成市监规〔2023〕6号**这一文号**未在任何参考资料中出现**。所有引用材料中提及的政策文号包括：

- 成经信办〔2023〕24号（《成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则》的印发通知）[1]
- 成经信发〔2023〕3号（《成都市关于进一步促进新型显示产业高质量发展的若干政策》）[2]
- 以及《成都市建设高品质科创空间政策细则》（未标注具体文号）[3][4]

但**没有任何资料提及“成市监规〔2023〕6号”**，也未说明其对应政策名称、发文机关或内容。

因此，**根据提供的资料无法回答此问题**。

── 来源 ──
  [1] 成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则 · 第1页
  [2] 成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则 · 第13页
  [3] 成都市建设高品质科创空间政策细则 · 第13页
  [4] 成都市建设高品质科创空间政策细则 · 第4页
(py312) PS D:\BaiduNetdiskDownload\week10检索增强生成\week10 检索增强生成RAG\week10 检索增强生成RAG\rag_annual_report> python src/rag_pipeline.py --query "成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则这个政策标题对于的政策内容是什么" --no-rerank
2026-07-08 22:33:58,201 [INFO] Loading faiss with AVX2 support.
2026-07-08 22:33:58,201 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 22:33:58,201 [INFO] Loading faiss.
2026-07-08 22:33:58,257 [INFO] Successfully loaded faiss.
2026-07-08 22:33:58,269 [INFO] FAISS 索引加载完成，共 98 条向量
D:\Anaconda_envs\envs\py312\Lib\site-packages\jieba\_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
2026-07-08 22:33:58,357 [INFO] 构建 BM25 索引（分词中，请稍候）...
Building prefix dict from the default dictionary ...
2026-07-08 22:33:58,357 [DEBUG] Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
2026-07-08 22:33:58,358 [DEBUG] Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
Loading model cost 0.610 seconds.
2026-07-08 22:33:58,968 [DEBUG] Loading model cost 0.610 seconds.
Prefix dict has been built successfully.
2026-07-08 22:33:58,969 [DEBUG] Prefix dict has been built successfully.
2026-07-08 22:33:59,076 [INFO] BM25 索引完成
2026-07-08 22:33:59,969 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 22:33:59,974 [INFO] 向量召回: 10 条，最高分=0.730
2026-07-08 22:33:59,976 [INFO] BM25 召回: 10 条，RRF 后: 20 条
2026-07-08 22:33:59,976 [INFO] 最终使用 4 条上下文，最高向量相似度=0.730
2026-07-08 22:34:10,722 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则这个政策标题对于的政策内容是什么
============================================================

《成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则》是成都市为推动新型显示产业高质量发展而制定的配套操作性文件，其核心内容包括对投资驱动项目和关键技术攻关项目的分类支持政策。

具体政策内容如下：

1. **投资驱动项目支持**
   - 分为两类：
     - **第Ⅰ类**：固定资产投入达10亿元以上的项目，按投入规模（10亿、50亿、100亿元）分别给予800万元、900万元、1000万元奖励[1]；
     - **第Ⅱ类**：固定资产投入在1亿元以上、10亿元以下，且属于新型显示关键材料及组件（如玻璃基板、柔性显示基膜、掩膜版、偏光片、触控模组、电子气体、液晶材料、发光材料、光电芯片等）或关键设备及零部件（如曝光、显影、蚀刻、离子注入、巨量转移、气相沉积、蒸镀、检测、封装设备等）制造项目，按投入额（1亿、1.5亿、2亿、3亿、5亿、8亿元）分档给予350万至700万元不等的奖励[1]。

2. **关键技术攻关项目支持**
   - 同样分为两类：
     - **第Ⅰ类**：自主研制新型显示关键材料、器件、超高清视频设备（如4K/8K芯片、编辑系统、摄录存储设备、放映平台）、大功率激光器、光刻机、刻蚀机、硅光芯片等，年销售收入500万元以上，按研发投入的20%给予最高300万元补助[3]；
     - **第Ⅱ类**：自主研制掩膜版、彩色滤光片、偏光片、驱动IC等技术（工艺），年销售收入1000万元以上，按研发投入的16%给予最高300万元补助[3]。

此外，该细则还明确了申报条件、支持标准及所需申报材料，如项目资金申报书、营业执照、合同证明、首次生产服务及实际收入证明等[3]。

── 来源 ──
  [1] 成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则 · 第3页
  [2] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件2.16 > 重点行业联盟链评价指标 · 第105页
  [3] 成都市关于进一步促进新型显示产业高质量发展的若干政策实施细则 · 第6页
  [4] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件2.15 > 成都市建设国家区块链创新应用综合试点专项政策项目推荐汇总表 > 推荐单位（单位盖章）： > 联系人： > 填报时间： · 第105页
(py312) PS D:\BaiduNetdiskDownload\week10检索增强生成\week10 检索增强生成RAG\week10 检索增强生成RAG\rag_annual_report> python src/rag_pipeline.py --query "这些政策中最高补助多少钱" --no-rerank
2026-07-08 22:34:44,817 [INFO] Loading faiss with AVX2 support.
2026-07-08 22:34:44,818 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 22:34:44,818 [INFO] Loading faiss.
2026-07-08 22:34:44,840 [INFO] Successfully loaded faiss.
2026-07-08 22:34:44,852 [INFO] FAISS 索引加载完成，共 98 条向量
D:\Anaconda_envs\envs\py312\Lib\site-packages\jieba\_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
2026-07-08 22:34:44,940 [INFO] 构建 BM25 索引（分词中，请稍候）...
Building prefix dict from the default dictionary ...
2026-07-08 22:34:44,940 [DEBUG] Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
2026-07-08 22:34:44,940 [DEBUG] Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
Loading model cost 0.601 seconds.
2026-07-08 22:34:45,541 [DEBUG] Loading model cost 0.601 seconds.
Prefix dict has been built successfully.
2026-07-08 22:34:45,541 [DEBUG] Prefix dict has been built successfully.
2026-07-08 22:34:45,643 [INFO] BM25 索引完成
2026-07-08 22:34:46,402 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 22:34:46,406 [INFO] 向量召回: 10 条，最高分=0.649
2026-07-08 22:34:46,407 [INFO] BM25 召回: 10 条，RRF 后: 17 条
2026-07-08 22:34:46,407 [INFO] 最终使用 4 条上下文，最高向量相似度=0.649
2026-07-08 22:34:51,210 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：这些政策中最高补助多少钱
============================================================

根据参考资料，这些政策中**最高补助金额为500万元**。

具体依据如下：

- 对高品质科创空间的改造及公共技术设备设施购置等费用，给予**最高不超过500万元**的经费补贴[3]；
- 对新引进的国内外知名创新型孵化器运营机构入驻科创空间建设创新创业载体，也给予**最高不超过500万元**的一次性经费补贴[3]。

其他政策中的最高补助金额均低于此标准，例如：
- 区块链产业国际性、专业化活动补贴最高**150万元**[1]；
- 区块链对接、协作项目活动补助最高**20万元**[2]；
- 支持组建国家重点实验室等国家级创新平台，一次性配套资助**300万元**[3]；
- 人才基地建设补贴最高**100万元**[4]。

因此，综合全部参考资料，**最高补助金额为500万元**[3]。

── 来源 ──
  [1] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件上传并提交。活动情况报告请同时提供Word 版。 > 五、请准备好申报材料及附属材料原件以备项目审核、审计 > 七、其他相关证 明材料。 > 区块链产业国际性、专业化活动补贴项目申报表 > 6. □其他 · 第97页
  [2] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件上传并提交。活动情况报告请同时提供Word 版。 > 五、请准备好申报材料及附属材料原件以备项目审核、审计 > 七、其他相关证 明材料。 > 区块链对接、协作项目活动补助项目申报表 > 6. □其他 · 第102页
  [3] 成都市建设高品质科创空间政策细则 · 第10页
  [4] 成都市建设高品质科创空间政策细则 · 第18页
(py312) PS D:\BaiduNetdiskDownload\week10检索增强生成\week10 检索增强生成RAG\week10 检索增强生成RAG\rag_annual_report> python src/rag_pipeline.py --query "最高补助金额的政策标题是什么" --no-rerank
2026-07-08 22:35:46,880 [INFO] Loading faiss with AVX2 support.
2026-07-08 22:35:46,881 [INFO] Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
2026-07-08 22:35:46,881 [INFO] Loading faiss.
2026-07-08 22:35:46,903 [INFO] Successfully loaded faiss.
2026-07-08 22:35:46,915 [INFO] FAISS 索引加载完成，共 98 条向量
D:\Anaconda_envs\envs\py312\Lib\site-packages\jieba\_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
2026-07-08 22:35:47,000 [INFO] 构建 BM25 索引（分词中，请稍候）...
Building prefix dict from the default dictionary ...
2026-07-08 22:35:47,001 [DEBUG] Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
2026-07-08 22:35:47,001 [DEBUG] Loading model from cache C:\Users\65467\AppData\Local\Temp\jieba.cache
Loading model cost 0.597 seconds.
2026-07-08 22:35:47,599 [DEBUG] Loading model cost 0.597 seconds.
Prefix dict has been built successfully.
2026-07-08 22:35:47,599 [DEBUG] Prefix dict has been built successfully.
2026-07-08 22:35:47,700 [INFO] BM25 索引完成
2026-07-08 22:35:48,423 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1 200 OK"
2026-07-08 22:35:48,426 [INFO] 向量召回: 10 条，最高分=0.632
2026-07-08 22:35:48,427 [INFO] BM25 召回: 10 条，RRF 后: 16 条
2026-07-08 22:35:48,427 [INFO] 最终使用 4 条上下文，最高向量相似度=0.632
2026-07-08 22:35:51,141 [INFO] HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"

============================================================
问题：最高补助金额的政策标题是什么
============================================================

最高补助金额的政策标题是：**2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南**。

依据该指南附件2中“区块链产业国际性、专业化活动补贴项目申报表”规定，该项补贴“按实际投入的50%，最高不超过150万元”[1]，为所列两类活动中最高补助限额（另一类“区块链对接、协作项目活动补助”最高为20万元[2]）。

因此，对应最高补助金额（150万元）的政策标题即为上述指南名称。

── 来源 ──
  [1] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件上传并提交。活动情况报告请同时提供Word 版。 > 五、请准备好申报材料及附属材料原件以备项目审核、审计 > 七、其他相关证 明材料。 > 区块链产业国际性、专业化活动补贴项目申报表 > 6. □其他 · 第97页
  [2] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件上传并提交。活动情况报告请同时提供Word 版。 > 五、请准备好申报材料及附属材料原件以备项目审核、审计 > 七、其他相关证 明材料。 > 区块链对接、协作项目活动补助项目申报表 > 6. □其他 · 第102页
  [3] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件上传并提交。活动情况报告请同时提供Word 版。 > 五、请准备好申报材料及附属材料原件以备项目审核、审计 > 七、其他相关证 明材料。 · 第97页
  [4] 2023年成都市建设国家区块链创新应用综合性试点专项政策项目申报指南 · 附件2 > 附件上传并提交。活动情况报告请同时提供Word 版。 > 五、请准备好申报材料及附属材料原件以备项目审核、审计 > 七、其他相关证 明材料。 · 第102页
"""

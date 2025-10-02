- [LlamaIndex](#LlamaIndex)
- Encoder-only, encoder-decoder,decoder-only
- Zero / One / Few shot
- LangChain
- Runnable in LCEL
- LangGraph
- Llama / Llama-2 / Llama-3
- Tokens / tokenization
- NIM
- Instruction tuning / alignment
- Embeddings
- Stochastic Parrot
- Auto-regressive forecasting
- Agentics (e.g. ReAct) and tools
- GPT2, GPT3, ChatGPT, GPT4
- Quantization
- Whisper Model
- T5 model
- Transformers and self/cross attention
- Query / Key / Value
- Masking / MLM
- Hugging Face
- Foundation Model
- BERT
- RAG
- Hallucinations
- CLIP / ViT / VLM / diffusion
  


---

# LlamaIndex {#ndex}

**是什麼**
資料接入與檢索增強生成（RAG）框架。把檔案/DB/API轉成索引（向量、樹、清單…），提供 Query Engine 與 Router，讓 LLM 能精準取用外部知識。

**為什麼重要**
解決「私有知識檢索、資料孤島、長文件摘要」等企業級需求；與 LangChain/LangGraph、各家向量庫（FAISS/PGVector）整合順暢。

**例子（Python）**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
docs = SimpleDirectoryReader("docs").load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5)
print(query_engine.query("合約的解約條款？"))
```

**面試要點**

* 何時用：長文件問答、法規/合約檢索、產品手冊客服。
* 指標：Recall@k、答案來源引用（source grounding）、延遲（latency）。

---

# 🔧 Encoder-only / Encoder-decoder / Decoder-only

**是什麼**

* **Encoder-only**（如 BERT）：善理解/分類/檢索。
* **Encoder-decoder**（如 T5、BART）：輸入→輸出轉換（翻譯、摘要）。
* **Decoder-only**（如 GPT 家族、Llama）：自回歸生成。

**例子**

* 情緒分類→BERT；新聞摘要→T5；對話/程式生成→GPT/Llama。

**面試要點**
用任務類型對映結構：理解→Encoder、轉換→Enc-Dec、自由生成→Dec-only。

---

# 🎯 Zero / One / Few-shot

**是什麼**
描述推理時示例數量：0/1/少量。Few-shot 透過示例模板提高穩定性。

**例子（Prompt 片段）**

```
You are a classifier. 
Examples:
Q: "Great battery life" → Positive
Q: "Terrible camera" → Negative
Now classify: "Screen is okay but slow"
```

**面試要點**

* Few-shot > Zero-shot 的情況：類別微妙、語域特定。
* 與「指令微調（SFT）」相比是推理時策略，不是再訓練。

---

# 🔗 LangChain

**是什麼**
開源框架，用來組裝 LLM、工具、記憶體與資料來源；提供 Chains、Agents、工具介面與 LCEL。

**例子（檢索 + LLM）**

```python
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
llm = ChatOpenAI(model="gpt-4o-mini")
retriever = FAISS.load_local("idx").as_retriever(k=4)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
qa("合約解約條文重點是什麼？")
```

**面試要點**

* 何時用 Agent：需工具選擇/多步計畫/動態決策。
* 何時用 Chain：流程固定、可管線化。

---

# ⚙️ Runnable in LCEL

**是什麼**
LCEL（LangChain Expression Language）讓鏈條能以「函式管線」組裝；`Runnable` 物件支援 `.invoke()`、`.stream()`。

**例子**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
prompt = ChatPromptTemplate.from_messages([("user","Summarize: {text}")])
chain = prompt | ChatOpenAI(model="gpt-4o-mini")
chain.invoke({"text": "Long paragraph..."})
```

**面試要點**

* 優點：組合安全、易測試、可觀測（tracing）。
* 與傳統 Chains 比：更像 Unix pipes。

---

# 🌐 LangGraph

**是什麼**
基於狀態機/圖的 Agent 框架；支援分支、回圈、錯誤恢復、長對話記憶。

**例子（簡化圖）**

```
[plan] -> [search] -> [reflect] -> {done? yes: [answer], no: back to [plan]}
```

**面試要點**
用在多步決策、需要回溯或人機協作的代理。

---

# 🦙 LLaMA / LLaMA-2 / LLaMA-3

**是什麼**
Meta 開源 Decoder-only 家族：參數段（7B/13B/70B…）、聊天/指令版（-Instruct）、多語種。

**例子**

* Llama-3-8B-Instruct：邊緣/本地推論。
* Llama-3-70B：高品質聊天與工具使用。

**面試要點**

* 與 GPT 系列差異：開源可自訓/量化，上線成本更可控。

---

# 🧩 Tokens / Tokenization

**是什麼**
Token 是模型的最小處理單位；BPE/WordPiece 將字串切成子詞片段。成本與長度限制以 token 計。

**例子**
"unaffordable" → `["una", "ff", "ordable"]`（示意）
中文常以字/字片段分割。

**面試要點**

* 中英文 token 比例差→估算成本/上下文長度要留意。

---

# 🚀 NIM（NVIDIA Inference Microservice）

**是什麼**
NVIDIA 的推論微服務封裝：模型容器 + Triton/性能最佳化 + 穩定 API。企業可即插即用布署 LLM/多模態。

**面試要點**

* 優勢：效能、可觀測、企業支援；與 RAG/向量DB/GuardRails 串接。

---

# 🧭 Instruction Tuning / Alignment

**是什麼**

* **Instruction Tuning（SFT）**：用指令/問答資料微調，讓模型遵循指令。
* **Alignment**：用 RLHF/DPO 等讓輸出符合人類偏好與安全。

**例子**
SFT：蒐集「指令→理想答案」對；RLHF：標註比較 A/B，學習偏好。

**面試要點**

* 何時只做 SFT？需求簡單、資料乾淨。
* 何時做 RLHF/DPO？需要價值對齊與禮貌、拒答邊界。

---

# 🔡 Embeddings

**是什麼**
把文字/圖片轉向量，供相似度（cosine/dot）檢索、聚類、推薦、RAG。

**例子（cosine）**
`sim(a,b) = (a·b)/(|a||b|)`；查詢向量與文件向量最相近者即候選。

**面試要點**

* 維度/常模化/向量庫選擇（HNSW, IVF, PQ）。
* 新鮮度：定期重嵌或混合檢索（BM25+向量）。

---

# 🦜 Stochastic Parrot

**是什麼**
批評 LLM 僅是「統計鸚鵡」，在未理解語義下模仿語言分佈。

**面試要點**

* 回應方式：結合外部工具/RAG/程式執行，提升「可驗證性」。

---

# 🔮 Auto-regressive Forecasting

**是什麼**
自回歸：用前序步驟預測下一步（語言模型逐 token 生成、時間序列 AR/ARIMA）。

**例子**
`P(x_t | x_{<t})`；生成直到 `<eos>` 或達長度上限。

**面試要點**

* 優勢：簡潔、可線上生成；
* 代價：長序列暴增的計算（注意 KV Cache）。

---

# 🤖 Agentics（ReAct）與工具

**是什麼**
Agent 可規劃/選工具/反思；**ReAct** 模式：先推理（Reason）再行動（Act），並根據觀察（Observation）迭代。

**例子（結構示意）**

```
Thought: 需要查匯率
Action: 調用「匯率API」
Observation: USD/TWD = 32.1
Thought: 完成計算
Final Answer: ...
```

**面試要點**

* 加入「工具」類型：檢索、計算、SQL、程式執行、瀏覽。
* 風險：工具濫用/循環；用 LangGraph 設定節點/終止條件。

---

# 🧠 GPT-2 / GPT-3 / ChatGPT / GPT-4

**是什麼**

* **GPT-2**：早期 Decoder-only。
* **GPT-3**：175B，引爆少樣本學習。
* **ChatGPT（GPT-3.5）**：對話微調與對齊。
* **GPT-4**：更強推理、多模態。

**面試要點**

* 差異維度：參數量、對齊程度、多模態、工具使用能力。

---

# 🪶 Quantization（量化）

**是什麼**
把 FP32 權重壓低到 FP16/INT8/INT4 以降記憶體/加速推論。方法：PTQ（GPTQ、AWQ）、QAT（訓練期量化）。

**例子（bitsandbytes 4-bit）**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-instruct", quantization_config=bnb, device_map="auto")
```

**面試要點**

* 取捨：速度/記憶體 vs. 精度；有時需校正或分層量化。
* KV cache 也可量化（注意延遲/品質）。

---

# 🎙 Whisper Model

**是什麼**
OpenAI 多語言 ASR（語音轉文字）；抗噪好，支援翻譯。

**例子**

```python
import whisper
m = whisper.load_model("small")
result = m.transcribe("meeting.mp3", task="transcribe")
print(result["text"])
```

**面試要點**

* 延遲 vs 準確率：模型尺寸選擇；VAD/分段策略很關鍵。

---

# 📝 T5（Text-to-Text Transfer Transformer）

**是什麼**
把所有任務都表述成「文字→文字」，採 Encoder-decoder。Prompt 即任務描述（如：`translate English to German:`）。

**例子**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
inp = tok("summarize: "+ long_text, return_tensors="pt")
print(tok.decode(model.generate(**inp, max_new_tokens=80)[0], skip_special_tokens=True))
```

**面試要點**

* 優勢：任務統一；劣勢：長輸出成本高、訓練較複雜。

---

# ⚡ Transformers 與 Self/Cross Attention

**是什麼**
Transformer = 前饋+（自/交叉）注意力 + 殘差 & LayerNorm。

* **Self-Attention**：序列內互相關注。
* **Cross-Attention**：解碼器關注編碼器輸出。

**核心計算（可讀版）**
`scores = (Q K^T) / sqrt(d)` → `weights = softmax(scores)` → `output = weights V`
複雜度（self-attn）：`O(n^2 · d)`。

**面試要點**

* Multi-Head 的作用：分解注意力子空間，提升表徵能力。
* 長序列：用稀疏/線性注意力或滑窗、RoPE/ALiBi。

---

# 🔑 Query / Key / Value（Q/K/V）

**是什麼**

* Q：我在問什麼
* K：每個位置的「特徵索引」
* V：要取回的內容
  權重看 Q 與 K 的相似度，然後加權 V。

**小例子**
「他」在句中需靠前文「小明」的 K 來決定指涉，V 提供具體語義。

---

# 🪞 Masking / MLM

**是什麼**

* **Causal Mask**：防看未來（Decoder-only 生成）。
* **MLM（Masked LM）**：遮蔽部分 token 讓模型預測（BERT 預訓練）。

**例子**
句子：「我今天去 [MASK]」→ 模型預測「上班/學校」。
Causal mask：上三角遮罩，位置 *t* 只能看到 `≤ t`。

**面試要點**

* BERT → MLM；GPT → Causal LM；T5 → Span Corruption。

---

# 🤗 Hugging Face

**是什麼**
社群平台 + 工具：`transformers`、`datasets`、`evaluate`、Hub、Inference Endpoints。

**例子**

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")
clf("Battery life is surprisingly good.")
```

**面試要點**

* 企業上線：私有 Model Hub、Endpoint、權限控管、審計。

---

# 🏛 Foundation Model

**是什麼**
大規模預訓練、可遷移到多任務的「基座模型」（語言、視覺、語音、多模態）。

**面試要點**

* 二階段：預訓練→調適（SFT/RLHF/指令資料/工具化）。

---

# 🐝 BERT

**是什麼**
雙向 Encoder-only；預訓練任務：**MLM + NSP**（或替代任務）。擅長理解：分類、抽取、NLI、檢索編碼器。

**例子**
問答系統中做段落檢索（Bi-Encoder）或抽取答案（Span Extraction）。

**面試要點**

* 與 GPT 差異：BERT 強理解、GPT 強生成。

---

# 🔍 RAG（Retrieval-Augmented Generation）

**是什麼**
先檢索→把證據與問題一起丟給 LLM→生成含引用的答案。

**示意**

1. 查詢嵌入→向量庫 top-k
2. 合成 Prompt（含引文）
3. LLM 生成並引用段落

**面試要點**

* 常見優化：重排序（Rerank）、多跳檢索、段落分塊、路由（query routing）、答案驗證。
* KPI：來源覆蓋率、事實性（faithfulness）、精確率/召回率。

---

# 🧪 Hallucinations（幻覺）

**是什麼**
可信口吻但內容錯；常因檢索缺失、知識過時、過度補全。

**緩解**
RAG、工具執行/程式驗證、限制型提示、置信度/不確定性回報、人審。

**面試要點**

* 面試答法：強調「可驗證管道」與防呆（Guardrails）。

---

# 🖼 CLIP / ViT / VL

**是什麼**

* **CLIP**：對齊圖文嵌入（對比式學習）。
* **ViT**：把影像切 Patch 當 Token，餵入 Transformer。
* **VL（Vision-Language）**：圖文多模態（如 BLIP、LLaVA）。

**例子**

* 零樣本分類：以文字提示「一張狗的照片」去比對圖片嵌入。
* 檢索：文字找圖/圖找文。

**面試要點**

* CLIP 的關鍵：共享嵌入空間（文本/圖像對齊）。
* ViT 的關鍵：Patch Embedding + 位置編碼（常用 RoPE）。

---

## 附錄：兩個綜合小範例

### 1) **RAG 最小可行系統（LangChain + 向量庫）**

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 建庫
docs = open("policy.txt","r",encoding="utf-8").read()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(docs)
emb = OpenAIEmbeddings()
db = FAISS.from_texts(chunks, emb)

# 檢索 + 生成
retriever = db.as_retriever(search_kwargs={"k":4})
llm = ChatOpenAI(model="gpt-4o-mini")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
res = qa("公司筆電遺失流程？")
print(res["result"])
for s in res["source_documents"]:
    print("source:", s.metadata)
```

### 2) **ReAct 代理（結構示意，不含真 API）**

```python
def agent(query):
    plan = f"為回答「{query}」需要：1) 搜索最新數據 2) 計算 3) 彙整"
    obs1 = call_tool("web_search", query)
    obs2 = call_tool("calculator", parse(obs1))
    return synthesize_answer(query, obs1, obs2)
```

---

## 快速口試錦囊

* **先定義，再說用途，最後給例子**（DEU：Definition→Employment→Use case）。
* 論模型：**結構差異 → 訓練任務 → 典型應用 → 取捨**。
* 論系統：**RAG/Agent** 強調「可驗證、可觀測、可維運」。
* 論品質：談 **幻覺治理、延遲/吞吐、指標**（EM/F1、BLEU、Rouge、Faithfulness）。

---

如果你要，我可以把本文件**轉成投影片大綱**或**加上中英並列**版本，或加入**你面試會被問到的專案實例**（如校務/電商/產學專案場景）。

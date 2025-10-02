- [LlamaIndex](#llamaindex)
- [Encoder-only / Encoder-decoder / Decoder-only](#encoder-only--encoder-decoder--decoder-only)
- [Zero / One / Few-shot](#zero--one--few-shot)
- [LangChain](#langchain)
- [Runnable in LCEL](#runnable-in-lcel)
- [LangGraph](#langgraph)
- [Llama / Llama-2 / Llama-3](#llama--llama-2--llama-3)
- [Tokens / Tokenization](#tokens--tokenization)
- [NIM](#nim)
- [Instruction Tuning / Alignment](#instruction-tuning--alignment)
- [Embeddings](#embeddings)
- [Stochastic Parrot](#stochastic-parrot)
- [Auto-regressive Forecasting](#auto-regressive-forecasting)
- [Agentics (ReAct) \& Tools](#agentics-react--tools)
- [GPT2 / GPT3 / ChatGPT / GPT4](#gpt2--gpt3--chatgpt--gpt4)
- [Quantization](#quantization)
- [Whisper Model](#whisper-model)
- [T5 Model](#t5-model)
- [Transformers \& Self/Cross Attention](#transformers--selfcross-attention)
- [Query / Key / Value](#query--key--value)
- [Masking / MLM](#masking--mlm)
- [Hugging Face](#hugging-face)
- [Foundation Model](#foundation-model)
- [BERT](#bert)
- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
- [Hallucinations](#hallucinations)
- [CLIP / ViT / VLM / Diffusion](#clip--vit--vlm--diffusion)
  
---

| 名稱                     | 英文解釋                                  | 中文摘要                 | 重點                          |
| ---------------------- | ------------------------------------- | -------------------- | --------------------------- |
| **LlamaIndex**         | Data framework for RAG (ex GPT Index) | 資料接入與索引化，支援多來源檢索     | RAG、索引、多來源                  |
| **Encoder-only**       | Only encoder (e.g. BERT)              | 單純理解/分類模型            | 句向量、語義理解                    |
| **Encoder-decoder**    | Encoder→Decoder (e.g. T5)             | 翻譯、摘要等輸入→輸出任務        | Seq2Seq、跨序列                 |
| **Decoder-only**       | Only decoder (e.g. GPT)               | 自回歸生成                | 長文本生成                       |
| **Zero/One/Few-shot**  | Prompting with 0/1/few examples       | 提示工程技術               | Few-shot>One-shot>Zero-shot |
| **LangChain**          | LLM app framework                     | 串接模型、記憶體、工具          | Chains、Agents               |
| **Runnable (LCEL)**    | Composable unit in LCEL               | 可組合的可執行單元            | invoke/stream               |
| **LangGraph**          | DAG-based agent workflow              | 圖式 Agent 狀態機         | DAG、循環、條件                   |
| **Llama-1/2/3**        | Meta’s open LLMs                      | 開源、商用友好、持續升級         | 私有部署                        |
| **Tokenization**       | Split text to tokens                  | 字詞分片，轉換為ID           | BPE、token成本                 |
| **NIM**                | NVIDIA Inference Microservice         | 模型推理容器化              | Triton、TensorRT             |
| **Instruction tuning** | Fine-tune on task instructions        | 指令微調                 | 提高服從指令能力                    |
| **Alignment**          | Align model to human values           | 模型價值/安全調校            | RLHF                        |
| **Embeddings**         | Text→vector                           | 語意向量                 | 檢索、相似度                      |
| **Stochastic Parrot**  | LLM = random pattern parrot           | 隨機鸚鵡比喻               | 缺乏理解                        |
| **Auto-regressive**    | Predict next token                    | 自回歸序列生成              | GPT核心原理                     |
| **Agentics / ReAct**   | Reason+Act with tools                 | 智能體推理+工具調用           | Agent、工具鏈                   |
| **GPT2/3/ChatGPT/4**   | Generative Pretrained Transformers    | GPT 系列發展史            | Decoder-only、多模態(GPT-4)     |
| **Quantization**       | Compress FP32→INT8/FP8                | 壓縮加速推論               | GPTQ/QAT/PTQ                |
| **Whisper**            | ASR model by OpenAI                   | 自動語音辨識、多語言           | 抗雜訊、轉錄                      |
| **T5**                 | Text-to-Text Transfer Transformer     | 任務統一化                | Encoder-decoder             |
| **Transformers**       | Attention-based architecture          | 取代RNN/CNN            | Self/Cross Attention        |
| **Q/K/V**              | Query/Key/Value                       | 注意力三元素               | softmax(QKᵀ)V               |
| **Masking/MLM**        | Masked Language Modeling              | 隱藏填補訓練               | BERT 預訓練                    |
| **Hugging Face**       | Open ML platform                      | Transformers套件、模型Hub | 生態完整                        |
| **Foundation Model**   | Large pre-trained adaptable model     | 基礎大模型                | 通用+微調                       |
| **BERT**               | Bidirectional Encoder                 | 雙向語義理解               | MLM+NSP                     |
| **RAG**                | Retrieval-Augmented Generation        | 檢索輔助生成               | 外部知識補充                      |
| **Hallucinations**     | Confident wrong outputs               | 虛構錯誤內容               | RAG可減少                      |
| **CLIP**               | Align text & image embeddings         | 圖文對齊                 | 解決 modality gap             |
| **ViT**                | Vision Transformer                    | 影像特徵抽取               | Transformer in CV           |
| **VLM**                | Vision-Language Model                 | 多模態模型                | 圖文推理                        |
| **Diffusion**          | Denoising generative model            | 去噪生成影像               | Stable Diffusion            |


---


## LlamaIndex

**解釋**：
LlamaIndex（原名 GPT Index）是一個 **資料接入與檢索增強生成 (RAG)** 框架，用來把結構化或非結構化資料（文件、資料庫、API）整理成可供大型語言模型 (LLM) 高效檢索的索引。它支援多種索引結構（Vector Store Index、Tree Index、Keyword Table Index 等）。

**重點摘要**：

* RAG 解決長文本記憶限制
* 支援多資料來源 (PDF、SQL、API)
* 內建查詢管線（Query Engine、Retriever）
* 與 LangChain、OpenAI API 相容

---

## Encoder-only / Encoder-decoder / Decoder-only

**解釋**：

* **Encoder-only**：僅使用編碼器，適合理解與分類（如 BERT）。
* **Encoder-decoder**：先編碼輸入，再解碼輸出，適合翻譯、摘要（如 T5）。
* **Decoder-only**：只用解碼器，適合自回歸生成（如 GPT 系列）。

**重點摘要**：

* Encoder-only：理解、embedding
* Encoder-decoder：輸入到輸出轉換
* Decoder-only：長文本生成、聊天

---

## Zero / One / Few-shot

**解釋**：

* **Zero-shot**：模型完全靠預訓練知識回答，無示例。
* **One-shot**：提供一個範例後再回答。
* **Few-shot**：給數個範例，幫助模型理解任務。

**重點摘要**：

* 提示工程核心技巧
* Few-shot > One-shot > Zero-shot（精度通常較高）

---

## LangChain

**解釋**：
一個 **構建 LLM 應用** 的框架，提供「Chain」與「Agent」抽象，結合模型、工具、記憶體、資料庫等。

**重點摘要**：

* 模組化組件 (LLM、Prompt、Memory)
* 支援 Agentic Workflow
* 與向量資料庫、RAG 整合良好

---

## Runnable in LCEL

**解釋**：
LangChain Expression Language (LCEL) 的核心概念。Runnable 是可組合的計算單元，透過 `.invoke()` 或 `.stream()` 執行，可拼接成複雜流程。

**重點摘要**：

* 提供同步/非同步呼叫
* 支援 Pipeline 式組合
* 是 LangChain 新一代 API 核心

---

## LangGraph

**解釋**：
LangChain 團隊推出的 **圖式工作流框架**，使用有向圖（DAG）來設計 LLM Agent 流程，可視覺化 Agent 狀態。

**重點摘要**：

* Agent 狀態機 (state machine)
* DAG + 記憶體，支援循環、條件分支
* 適合大型多工具 Agent

---

## Llama / Llama-2 / Llama-3

**解釋**：
Meta 發布的開源 LLM 系列。

* **Llama**：2023 首版，研究導向。
* **Llama-2**：釋出商用授權，支援聊天模型。
* **Llama-3**：2024 發布，支援更長上下文、對話強化。

**重點摘要**：

* 開源 + 商用友好
* Prompt-following 能力增強
* 在私有部署中受歡迎

---

## Tokens / Tokenization

**解釋**：
Token 是模型處理的最小單位（可能是詞片段、子詞、符號）。Tokenization 將文字轉成模型可理解的 ID 序列。

**重點摘要**：

* BPE / WordPiece 常用
* Token 數量影響推理成本與上下文長度

---

## NIM

**解釋**：
NVIDIA Inference Microservice，一種 **封裝推理服務** 的容器化解決方案，用於部署 LLM 或多模態模型。

**重點摘要**：

* 雲端/本地可快速部署
* 支援 Triton、TensorRT-LLM
* 適合企業級 AI 部署

---

## Instruction Tuning / Alignment

**解釋**：

* **Instruction tuning**：用指令-回應資料微調模型，提升遵循指令能力。
* **Alignment**：讓模型行為符合人類價值與期望，通常包含 RLHF。

**重點摘要**：

* Instruction tuning → 服從指令
* Alignment → 安全、符合人類偏好

---

## Embeddings

**解釋**：
把文字或其他資料轉成高維向量，用於相似度搜尋、聚類、語義檢索。

**重點摘要**：

* 用於向量資料庫 + RAG
* Cosine similarity / dot product

---

## Stochastic Parrot

**解釋**：
比喻 LLM「像隨機鸚鵡」，僅依據機率分布複製語言樣式，沒有真正理解。

**重點摘要**：

* 強調語言模型缺乏真實推理
* 來源：Emily Bender 論文 (2021)

---

## Auto-regressive Forecasting

**解釋**：
以序列前部分預測下一步輸出（如 GPT 根據先前 token 生成下一 token）。

**重點摘要**：

* 自回歸序列建模
* 用於語言生成、時間序列預測

---

## Agentics (ReAct) & Tools

**解釋**：

* **Agentics**：LLM + 工具使用的智能體設計方法。
* **ReAct**：Reason + Act 策略，模型先思考再調用工具。

**重點摘要**：

* 工具調用、API 連接
* ReAct：可推理 + 執行

---

## GPT2 / GPT3 / ChatGPT / GPT4

**解釋**：

* GPT2：2019，Transformer Decoder-only。
* GPT3：175B 參數，強化 few-shot。
* ChatGPT：GPT3.5 微調聊天。
* GPT4：多模態、推理能力更強。

**重點摘要**：

* 同屬 Decoder-only
* GPT4：多模態 + 安全性改進

---

## Quantization

**解釋**：
將高精度浮點權重（FP32）壓縮成低位元（INT8、FP8），減少記憶體與計算成本。

**重點摘要**：

* GPTQ、QAT、PTQ 技術
* 推論加速、成本降低，但精度可能下降

---

## Whisper Model

**解釋**：
OpenAI 發布的自動語音辨識 (ASR) 模型，支援多語言、強大抗雜訊。

**重點摘要**：

* End-to-end Transformer ASR
* 多語言、多口音
* 可轉錄+翻譯

---

## T5 Model

**解釋**：
Text-to-Text Transfer Transformer，Google 提出。把所有 NLP 任務轉成「輸入文字 → 輸出文字」。

**重點摘要**：

* Encoder-decoder
* 任務統一化設計
* 適合翻譯、摘要、問答

---

## Transformers & Self/Cross Attention

**解釋**：

* **Transformer**：以注意力機制為核心的架構。
* **Self-attention**：序列內部 token 互相關注。
* **Cross-attention**：解碼器關注編碼器輸出。

**重點摘要**：

* 取代 RNN/CNN
* Self-attention = 同序列
* Cross-attention = 跨序列

---

## Query / Key / Value

**解釋**：
注意力機制中：

* **Query (Q)**：當前 token 的查詢向量。
* **Key (K)**：上下文 token 的鍵向量。
* **Value (V)**：對應的資訊向量。

**重點摘要**：

* Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V
* Q 比對 K，取出加權 V

---

## Masking / MLM

**解釋**：

* **Masking**：隱藏部分輸入讓模型學習填補。
* **MLM**：Masked Language Model，如 BERT 用 [MASK] 預測。

**重點摘要**：

* 提升語義理解
* Encoder-only 常用訓練法

---

## Hugging Face

**解釋**：
最大開源 AI 平台，提供 Transformers 套件、模型 Hub、Datasets。

**重點摘要**：

* Transformers: LLM 主力套件
* 模型共享、生態完整

---

## Foundation Model

**解釋**：
在大量多樣化資料上預訓練，可適應多任務的大型基礎模型。

**重點摘要**：

* 通用 + 可微調
* LLM 與多模態模型的基礎

---

## BERT

**解釋**：
Bidirectional Encoder Representations from Transformers，Google 推出，雙向編碼器模型。

**重點摘要**：

* Encoder-only
* MLM + NSP 預訓練
* 革新 NLP 任務表現

---

## RAG (Retrieval-Augmented Generation)

**解釋**：
結合外部檢索與 LLM，先找相關內容再生成答案。

**重點摘要**：

* 解決上下文限制
* 提升準確度與可控性

---

## Hallucinations

**解釋**：
LLM 生成看似合理但實際錯誤或不存在的資訊。

**重點摘要**：

* 原因：缺乏真實知識、過度擬合語言模式
* RAG / 驗證管線可減少

---

## CLIP / ViT / VLM / Diffusion

**解釋**：

* **CLIP**：對齊圖像與文字嵌入，解決 modality gap。
* **ViT**：Vision Transformer，影像版 Transformer。
* **VLM**：Vision-Language Model，多模態理解。
* **Diffusion**：逐步去噪生成影像的模型（如 Stable Diffusion）。

**重點摘要**：

* CLIP → 圖文對齊
* ViT → 影像特徵抽取
* VLM → 多模態推理
* Diffusion → 文生圖主流技術




---



- [Retrieve Anything to Augment Large Language Models (RA-LLM, 2024)](#retrieve-anything-to-augment-large-language-models-ra-llm-2024)
- [PowerNorm: Rethinking Batch Normalization in Transformers (2003.07845)](#powernorm-rethinking-batch-normalization-in-transformers-200307845)
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper, OpenAI 2022)](#robust-speech-recognition-via-large-scale-weak-supervision-whisper-openai-2022)
- [Training Compute-Optimal Large Language Models (Chinchilla, 2203.15556)](#training-compute-optimal-large-language-models-chinchilla-220315556)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](#exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer-t5)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](#an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-vit)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG, 2020)](#retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-rag-2020)
- [Retrieval-Augmented Generation for Large Language Models: A Survey (2023/2024)](#retrieval-augmented-generation-for-large-language-models-a-survey-20232024)
---



| 縮寫 | 全名 | 中文解釋 | 主要出現的論文 |
|------|------|----------|----------------|
| **RAG** | Retrieval-Augmented Generation | 檢索增強生成 | - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks<br>- Retrieve Anything to Augment Large Language Models<br>- Retrieval-Augmented Generation for Large Language Models: A Survey |
| **ASR** | Automatic Speech Recognition | 自動語音辨識 | Robust Speech Recognition via Large-Scale Weak Supervision |
| **BN** | Batch Normalization | 批次正規化 | PowerNorm: Rethinking Batch Normalization in Transformers |
| **NSP** | Next Sentence Prediction | 下一句預測 | T5 論文提及 BERT 缺點（移除 NSP） |
| **LCEL** | LangChain Expression Language | LangChain 的工作流語法 | LangChain / LangGraph 框架 |
| **DAG** | Directed Acyclic Graph | 有向無環圖 | LangGraph 中用於 Agent 工作流 |

---

### 論文簡要摘要表

| 論文 | 年份 / 作者 | 核心貢獻 | 要點 |
|------|------------|---------|------|
| **Retrieve Anything to Augment Large Language Models** | 2024 | 擴展 RAG 到多模態與任意資料型態 | 支援文字、圖片、表格、程式碼檢索；減少幻覺 |
| **PowerNorm: Rethinking Batch Normalization in Transformers** | 2020 (2003.07845) | 改進 BN(和LN) 以提升 Transformer 訓練穩定性 | 動態調整 BN，適用小 batch；收斂更快 |
| **Robust Speech Recognition via Large-Scale Weak Supervision** (Whisper) | 2022 OpenAI | 大規模弱監督 ASR | 68 萬小時多語音資料；Encoder-Decoder；抗雜訊、多語言 |
| **Training Compute-Optimal Large Language Models** (Chinchilla) | 2022 (2203.15556) | 提出 Chinchilla scaling law | Token 數與參數需平衡；GPT-3 不最優；計算更高效 |
| **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** (T5) | 2019 Google | 統一 NLP 任務為 Text-to-Text | Encoder-Decoder；C4 資料集；遷移學習成功 |
| **An Image is Worth 16x16 Words (ViT)** | 2020 (2010.11929) | 用 Transformer 做影像分類 | 將圖像切成 16×16 patches；需大規模預訓練 |
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | 2020 Lewis et al. | 系統化提出 RAG | Dense Retriever + Generator；解決 LLM 記憶限制 |
| **Retrieval-Augmented Generation for Large Language Models: A Survey** | 2023/24 | RAG 全景回顧與分類 | 檢索策略、融合方法、挑戰與未來方向 |

---

## Retrieve Anything to Augment Large Language Models (RA-LLM, 2024)

* **發現 / 問題**
  * 現有 RAG 僅適合文字，對程式碼、圖片、表格等非文字知識難以檢索與整合；導致 LLM 生成內容缺乏多元資料支援，容易產生幻覺。
  * 通用 embedder 雖多任務但不是為 LLM 檢索增強優化。
  * 專任務 retriever 雖專業但缺乏泛化能力，難以應付多種檢索需求。
  * 兩者在檢索與 LLM 之間缺少「對最終生成效果」的對齊
* **概念**
  * 建立通用檢索層（Retrieve Anything），支援跨模態、多來源檢索，讓 LLM 能以更豐富的知識生成答案。
  * 論文不是只看「哪個 embedding 向量更好」，而是看這些 embedding 在 檢索 → 餵給 LLM → LLM 輸出效果 這整條 pipeline 上的增益（即「檢索增強效果」）。

* **作法**

  * 設計 **多模態索引器**（文字、圖像、表格、程式碼 embedding）。
  * 統一檢索 API，抽象不同資料來源。
  * 以 reranking + 格式化處理後餵給 LLM。

* **創新點**
  * 引入來自 LLM 輸出效果的「回報 (reward)」作為軟標籤，進行知識蒸餾 (distillation)；
  * 多任務訓練 + 指令微調 (instruction tuning)，讓 embedding 模型能對應不同檢索任務；
  * 選用 homogeneous in-batch negative sampling，使得負樣本更具鑑別性。
  
* **貢獻**

  * 從文字擴展到「任何資料型態」。
  * 降低幻覺，提供更準確知識。
  * 提供一個模組化檢索架構。

* **重點摘要**

  > 「RA-LLM 解決了傳統 RAG 無法處理非文字資料的問題，提出通用多模態檢索層，降低 LLM 幻覺並豐富回答來源。」
  >  “在這篇論文中，作者比較了傳統稀疏檢索（BM25）、Dense 检索器（如 DPR）與各類專任務 embedded retriever，指出這些方法在為 LLM 提供知識支持時各有限制。作者提出 LLM-Embedder 作為統一向量表示模型，通過融合多任務訓練、LLM 回饋信號與優化策略，在多場景 (知識增強、長上下文檢索、工具選取) 均顯著優於基線方法。”

  >| 模型 / 方法 | 所屬類型 | 特性 / 訓練方式 | 比較場景 / 指標 | 與 LLM-Embedder 的差距 / 表現優勢 |
  >|-------------|----------|-----------------|-----------------|-----------------------------------|
  >| **BM25 / Lexical retriever (稀疏檢索)** | 傳統檢索器 | 基於關鍵詞匹配 / 倒排索引 | 排序準確度 / 檢索相關性 | 在語義層面不夠強，對於需要理解上下文的檢索表現較弱 |
  >| **Dense retriever（如 DPR）** | 向量檢索器 | 將 query / passage embed 成向量做相似度比對 | 檢索召回率 / 精度 | 在許多情境下被用作基線，LLM-Embedder 針對 LLM 增強做優化可超越 |
  >| **Task-specific embedder / retrievers** | 專任務 embedding / 檢索器 | 為某一任務（如 QA、對話、記憶檢索）單獨訓練 embedding | 任務指標 (QA 精度、對話檢索準確性) | 雖在該任務可能表現較好，但缺乏通用性；LLM-Embedder 在多任務場景中更具泛化能力 |
  >| **LLM-Embedder**（作者提出的新方法） | 統一 embedding 模型 | 多任務訓練 + LLM 回饋 (reward distillation) + instruction fine-tuning + in-batch homogeneous negative sampling | 整體檢索增強效益 (knowledge enhancement, in-context retrieval, long-context support, 工具檢索等) | 在多種檢索增強場景中 **超越一般 / 專任務 embedder**，更具通用性與效能 |

---

## PowerNorm: Rethinking Batch Normalization in Transformers (2003.07845)

* **發現 / 問題**
  Transformer 用 BatchNorm(和LayerNorm) 容易在小 batch 或分布變動時不穩定，訓練表現差；LayerNorm 雖穩定但收斂速度較慢。

* **概念**
  重新設計 BN，使其能在 Transformer 中穩定運作，結合 BN 的高效收斂和 LN 的穩定性。

* **作法**

  * 提出 **Power Normalization**，調整 BN 的縮放與方差計算方式，減少高維特徵下的偏差。
  * 保持計算成本低，與現有 BN 兼容。

* **貢獻**

  * 改善 BN 在 NLP Transformer 的不穩定性。
  * 提高訓練收斂速度且穩定。
  * 可直接替換原有 BN。

* **重點摘要**

  > 「PowerNorm 改良了 BN，解決 Transformer 小 batch 或分布不穩定時的訓練問題，兼具收斂速度和穩定性。」


  >| 特性 | **Batch Normalization (BN)** | **Layer Normalization (LN)** | **PowerNorm** |
  >|------|-----------------------------|-----------------------------|---------------|
  >| **正規化維度** | 對 **整個 batch** 的同一特徵維度 | 對 **單一樣本** 的所有特徵維度 | 改良 BN，重新縮放以減少高維度偏差 |
  >| **數學公式** | $\displaystyle \mu_B^{(k)} = \frac{1}{m} \sum_{i=1}^{m} x_{i}^{(k)}$ <br> $\displaystyle \sigma_B^{(k)} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(x_{i}^{(k)}-\mu_B^{(k)})^2+\epsilon}$ <br> $\displaystyle \hat{x}_{i}^{(k)} = \frac{x_{i}^{(k)}-\mu_B^{(k)}}{\sigma_B^{(k)}}$ | $\displaystyle \mu_L = \frac{1}{H}\sum_{k=1}^{H}x^{(k)}$ <br> $\displaystyle \sigma_L = \sqrt{\frac{1}{H}\sum_{k=1}^{H}(x^{(k)}-\mu_L)^2+\epsilon}$ <br> $\displaystyle \hat{x}^{(k)}=\frac{x^{(k)}-\mu_L}{\sigma_L}$ | $\displaystyle \hat{x}= \frac{x - \mu_B}{(\sigma_B^p + \epsilon)^{1/p}}$ <br> *(p-次方歸一化，p 通常取 2 或其他值以改善穩定性)* |
  >| **依賴 batch size** | ✅ 需要大 batch 才穩定 | ❌ 與 batch size 無關 | ✅ 但對小 batch 更穩定 |
  >| **推論** | 需使用 **移動平均統計量** | 訓練與推論公式一致 | 與 BN 類似但更穩定 |
  >| **優點** | 收斂快，CV 中效果佳 | 與 batch 無關，NLP/Transformer 穩定 | 保留 BN 收斂優勢 + 小 batch 穩定 |
  >| **缺點** | 小 batch 退化、分佈轉換敏感 | 收斂略慢 | —— (主要為 BN 改進版) |
  >| **典型應用** | CNN、CV 模型 | NLP、Transformer | Transformer 小 batch / 分布變化場景 |


---

## Robust Speech Recognition via Large-Scale Weak Supervision (Whisper, OpenAI 2022)

* **發現 / 問題**
  傳統語音辨識依賴大量人工標註資料，昂貴且語言多樣性不足，模型在雜訊或新語言上表現差。

* **概念**
  利用網路上大量「低品質但廣泛覆蓋」的語音 + 轉錄資料，透過弱監督打造強健多語言 ASR。

* **作法**

  * 蒐集 680k 小時多語音資料（低成本弱監督）。
  * Transformer Encoder–Decoder 模型處理 Mel-spectrogram 輸入，decoder 自回歸生成文字。
  * 多任務訓練（轉錄 + 翻譯）。

* **貢獻**

  * 大規模弱監督成功用於 ASR。
  * 高魯棒性：抗雜訊、多語言支援佳。
  * 提供可泛化的開放式 ASR 模型。

* **重點摘要**

  > 「Whisper 發現人工標註限制了 ASR，改用大規模弱監督資料訓練 Encoder–Decoder，使模型更通用且抗雜訊。」


  >| 模型 | 訓練資料規模 | 架構 | 語言覆蓋 | 噪聲/口音魯棒性 | WER (LibriSpeech test-other) | 主要特點 |
  >|------|--------------|------|----------|-------------------|-----------------------------|----------|
  >| **Whisper Large** | **680k 小時**<br>(弱監督, 多語言) | Transformer Encoder–Decoder | 96+ 語言 | **極高** | **2.7%** | 大規模弱監督、多任務 (轉錄+翻譯)，跨語言與噪聲場景表現最佳 |
  >| Wav2Vec 2.0 Large (LV-60) | 60k 小時 (Libri-Light + LibriSpeech) | CNN + Transformer Encoder | 英語 | 中等 | 2.9% | 自監督預訓練，微調於 LibriSpeech，主要針對英語 |
  >| Conformer (ESPnet) | 約 1k 小時 (LibriSpeech) | CNN + Transformer Encoder | 英語 | 中等 | 2.1% *(特化英語)* | 混合 CNN + Self-attention，針對乾淨英語表現優秀，但多語/噪聲較弱 |
  >| SpeechStew | 23k 小時 (混合多英語資料集) | Transformer Encoder | 英語 | 中等偏低 | 3.0% | 將多個英語語料混合訓練，對英文較好但跨語言差 |
  >| Whisper Medium | 680k 小時 | Transformer Encoder–Decoder | 96+ | 高 | 3.0% | 較小模型但仍具多語言與噪聲穩健性 |
  >| Whisper Small / Base | 680k 小時 | Transformer Encoder–Decoder | 96+ | 高 | 4~5% | 輕量化版本，速度快但精度略降 |

    > **WER** = Word Error Rate (越低越好)


---

## Training Compute-Optimal Large Language Models (Chinchilla, 2203.15556)

* **發現 / 問題**
  GPT-3 等模型過度放大參數，但訓練 token 數不足，計算資源利用不佳，性能未達最優。

* **概念**
  提出 **計算最優化 scaling law**，用理論找出在固定算力下最佳的模型大小與 token 數配置。

* **作法**

  * 分析不同模型大小與資料量在相同 FLOPs 下的 loss。
  * 提出 **Chinchilla scaling law**：優先增加訓練 token，而非無限增加參數。

* **貢獻**

  * 給出 LLM 設計實用公式。
  * 顛覆「大參數必勝」觀念。
  * 同等算力下可訓練更高效模型。

* **重點摘要**

  > 「Chinchilla scaling law 發現 GPT-3 過大但 token 不足，提出計算最優訓練法則：平衡參數與 token 數量以最佳化效能。」
  > | 模型 | 參數數量 (Params) | 訓練 Token 數 | 訓練 FLOPs | 主要觀察 | 結論 |
  > |------|------------------|---------------|------------|----------|------|
  > | **GPT-3** | 175B | ~300B | ≈3.14×10²³ | 參數極大但訓練 token 不足，導致計算資源未充分利用 | 僅加大參數量並不足以提升效率 |
  > | **Gopher** | 280B | ~300B | ≈4.6×10²³ | 大模型但資料量仍偏少，損失下降速度不理想 | 過度追求參數規模會導致次優訓練 |
  > | **Chinchilla** | 70B | **1.4T** | ≈3.0×10²³ | **在同等 FLOPs 下，增加 token 而非參數，效果優於 GPT-3/Gopher** | 計算最優：**用更多 token 訓練較小模型，優於大參數少資料** |
  > | **Scaling Law (理論)** | — | — | — | 提出最佳化公式：<br>$N_\text{params} \propto C^{0.73}$、$N_\text{tokens} \propto C^{1.27}$ | 在固定算力 $C$ 下，應**多訓練資料而非無限增大參數** 
   >$C$ = 訓練 FLOPs 總量


---

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)

* **發現 / 問題**
  NLP 任務格式不統一導致模型遷移學習困難，需為每個任務重新設計輸入輸出。

* **概念**
  用一個「Text-to-Text」框架統一所有 NLP 任務，簡化微調與遷移。

* **作法**

  * 訓練 **encoder-decoder Transformer**。
  * 建立 **C4** 大型乾淨語料庫。
  * 把翻譯、問答、分類、摘要全轉成 text-to-text 格式。

* **貢獻**

  * 建立 NLP 任務統一框架。
  * 驗證大規模預訓練 + 微調的威力。
  * 推動之後 prompt-based 設計思路。

* **重點摘要**
  > 「T5 發現 NLP 任務輸入輸出不統一阻礙遷移，提出 text-to-text 框架，用 encoder-decoder 搭配大語料 C4，讓任務統一化並提升微調效率。」


  >| 模型 | 架構 | 預訓練任務 | 預訓練資料集 | 下游任務表現 (GLUE avg) | 主要特點 |
  >|------|------|------------|--------------|-------------------------|----------|
  >| **T5 (11B)** | **Encoder–Decoder** | Span Corruption (Masked LM 改良) | **C4 (750GB)** | **89.7** | 將所有 NLP 任務統一成 text-to-text，使用大規模乾淨語料；多任務遷移學習效果極佳 |
  >| **BERT-Large** | Encoder-only | MLM + NSP | BookCorpus + Wikipedia | 82.1 | 雙向編碼，僅適用於理解任務 (分類/QA)；不擅長生成 |
  >| **RoBERTa-Large** | Encoder-only | MLM (無 NSP) | BookCorpus + Wikipedia + CC-News + OpenWebText + Stories | 88.5 | 改進 BERT (更多資料、無 NSP、動態 masking)；理解任務表現佳 |
  >| **XLNet-Large** | Permutation LM | 自回歸排列 LM + 二向性 | Wiki + Books + Giga5 + ClueWeb | 88.4 | 混合自回歸 + 自編碼優勢；強於 BERT，訓練複雜 |
  >| **GPT-2 (1.5B)** | Decoder-only | 自回歸 LM | WebText (40GB) | 72.0 (GLUE 不佳) | 專注生成與少樣本學習，理解任務表現弱 |
  >| **UniLM v2** | Unified LM | MLM + Seq2Seq + 自回歸 | BookCorpus + Wikipedia + CC-News | 88.2 | 嘗試統一生成與理解；但規模與資料不如 T5 |
  >| **Transformer-XL** | Decoder-only (記憶延伸) | 自回歸 LM | Wiki + Books | — | 對長序列建模更好，但非專門針對下游 GLUE |

    > **GLUE avg** = General Language Understanding Evaluation 平均分數 (越高越好)

---

## An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)

* **發現 / 問題**
  CNN 長期主導影像分類，但缺乏全域注意力；Transformer 雖強但在影像上直接使用困難，尤其資料不足時易過擬合。

* **概念**
  將影像視為 token 序列，直接套用 Transformer 處理視覺任務。

* **作法**

  * 圖片切成 **16×16 patch**，每塊當 token。
  * 加位置編碼後輸入 Transformer Encoder。
  * 大規模預訓練（JFT-300M）再遷移到 ImageNet。

* **貢獻**

  * 打破 CNN 壟斷，開啟 Vision Transformer 時代。
  * 需要大資料集才能表現優異。
  * 驗證純注意力對影像可行。

* **重點摘要**

  > 「ViT 發現 CNN 缺乏全域建模，提出將影像切成 patch 當 token，用 Transformer Encoder 做影像分類，需大規模資料才能成功。」

  >| 模型 | 架構 | 預訓練資料集 | ImageNet Top-1 精度 | 訓練特點 / 優缺點 |
  >|------|------|-------------|---------------------|-------------------|
  >| **ViT-B/16** | **純 Transformer Encoder** | JFT-300M (3億張圖) | **77.9%** (ImageNet, fine-tune) | 首次將影像切成 16×16 patches 並用 Transformer；需要大量資料預訓練以避免過擬合 |
  >| **ViT-L/16** | 純 Transformer Encoder (更大) | JFT-300M | **81.1%** | 大型 ViT，預訓練後在 ImageNet fine-tune 可超越 ResNet；但計算量高 |
  >| **ResNet-152** | CNN (殘差網路) | ImageNet | 78.5% | 當時最佳 CNN；易訓練但缺乏全域建模能力 |
  >| **EfficientNet-B7** | CNN (複合縮放) | ImageNet | 84.3% | 參數/計算高效；但設計複雜、缺乏全域注意力 |
  >| **RegNetY-16GF** | CNN (可調架構) | ImageNet | ~80% | CNN 架構搜尋設計；訓練穩定，效能不及大型 ViT |
  >| **DeiT-B/16** *(後續改進)* | ViT + 知識蒸餾 | ImageNet (無需超大外部資料) | 81.8% | 改進訓練策略，降低對大規模外部資料依賴 |


---

## Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG, 2020)

* **發現 / 問題**
  純生成模型知識過時且有限，對知識密集 QA 容易幻覺。

* **概念**
  將檢索外部資料庫與生成模型結合，讓模型在生成前先取得外部事實。

* **作法**

  * **Retriever**：Dense Passage Retrieval (DPR) 找相關段落。
  * **Generator**：seq2seq（如 BART/T5）融合檢索內容產生答案。
  * End-to-end 訓練檢索與生成。

* **貢獻**

  * 首次系統化定義「RAG」架構。
  * 減少幻覺、提升 QA 準確度。
  * 知識更新可替換資料庫，實用性高。

* **重點摘要**

  > 「RAG 發現 LLM 記憶有限且易幻覺，提出檢索 + 生成的架構：先 DPR 檢索，再用 seq2seq 生成，改善知識密集任務表現。」

  >| 模型 | 類型 | 檢索方式 | 生成模型 | 訓練策略 | 指標 (NQ EM / F1) | 主要特點 |
  >|------|------|---------|---------|---------|------------------|---------|
  >| **BERT-Large** | Encoder-only | 無檢索 | 無生成 (分類式 QA) | Fine-tune on QA | ~33 / 41 | 僅依賴參數記憶知識，無法處理開放域新知識 |
  >| **REALM** | 檢索增強預訓練 | Dense (learned retriever) | Encoder-only (MLM) | End-to-end 更新 retriever 與 encoder | ~40 / 49 | Google 提出，檢索和預訓練結合，但推理生成能力有限 |
  >| **Fusion-in-Decoder (FiD)** | 檢索增強生成 | Dense (DPR) | BART / T5 Decoder | 將檢索段落直接拼接進 decoder | ~48 / 58 | 對多段上下文加權融合，生成效果佳 |
  >| **BART-Large** | Encoder–Decoder | 無檢索 | Seq2Seq 生成 | Fine-tune on QA | ~36 / 44 | 純生成但無法外部擴充知識，長期記憶有限 |
  >| **DPR (Dense Passage Retriever)** | Dense 檢索器 | Dense (dual-encoder) | 無生成 (extractive) | 專門學習檢索向量 | ~41 / 53 | 高效 dense retriever，但僅擷取段落，無生成能力 |
  >| **RAG-Sequence** | 檢索+生成 | Dense (DPR) | BART Decoder (Seq2Seq) | End-to-end：檢索結果串接輸入 decoder，自回歸逐步生成 | **44 / 56** | 在生成過程中每一步使用檢索段落，逐 token 利用外部知識 |
  >| **RAG-Token** | 檢索+生成 | Dense (DPR) | BART Decoder (Seq2Seq) | 每個 token 時動態融合檢索信息 (token-level fusion) | **45 / 57** | 提升對長文本的利用效率，生成更準確 |

    > NQ = Natural Questions dataset  
    > EM = Exact Match，F1 = F1-score (越高越好)

---

## Retrieval-Augmented Generation for Large Language Models: A Survey (2023/2024)

* **發現 / 問題**
  RAG 相關研究快速發展但分散，缺乏系統化整理與未來方向指引。

* **概念**
  對 RAG 進行全面分類與總結，包含檢索策略、融合方法、效能挑戰、未來趨勢。

* **作法**

  * 分析檢索：sparse / dense / hybrid / multi-modal。
  * 融合策略：concat、fusion-in-decoder、rerank、memory-augmented。
  * 探討挑戰：延遲、即時更新、評估標準。

* **貢獻**

  * 提供完整技術地圖與挑戰清單。
  * 展望多模態與動態記憶 RAG。
  * 幫助研究者快速定位技術。

* **重點摘要**

  > 「這篇 survey 系統整理 RAG，包含檢索、融合方法及未來挑戰，如多模態擴展、即時更新、降低延遲，是理解 RAG 全景的關鍵。」

  >| 類型 / 方法 | 所屬範式 | 核心策略 / 特性 | 優點 | 限制 / 弱點 |
  >|--------------|--------------------|----------------------------|-------|-------------------------------|
  >| **Naive RAG** | 基本檢索 + 拼接 | 檢索出文件後直接拼接到 prompt，再交給 LLM 生成 | 實現簡單、低複雜度 | 拼接過多可能超長、檢索噪聲影響生成 |
  >| **Advanced RAG** | 加強檢索 / 融合 / 重排 | 在檢索或融合環節做更多改進（如重排序 / 選擇 / 加權融合） | 檢索質量與生成質量提升 | 計算成本較高、設計較複雜 |
  >| **Modular RAG** | 模組化／可插拔 | 檢索、融合、生成等模組可以替換 / 插拔 | 擴展性強、可自訂每模組 | 各模組整合難，模組間協調成本高 |
  >| **Retriever-level 方法** | 如 Dense / Hybrid / Sparse 检索策略 | 在檢索器設計上下功夫（比如用密集 embedding、混合檢索、重排等） | 檢索效果更好、召回率與精度提升 | 必須同步優化檢索與生成模組 |
  >| **Generation-level 融合策略** | Concat、Fusion-in-Decoder、Token-level 融合等 | 決定如何把檢索結果融合進 LLM：整段拼接 vs 在 decoder 層融合 vs token 級融合 | 在融合方式上可權衡效率與效果 | 融合太複雜可能造成模型不穩定或過擬合 |
  >| **Augmentation 方法** | 如 prompt 擴充、回饋重排序、知識校正等 | 在檢索或生成前後加入額外步驟來改善效果 | 增強生成可信度、降低幻覺 | 增加系統複雜度與延遲 |






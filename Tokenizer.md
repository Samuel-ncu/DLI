# Tokenize（斷詞/分詞/切字）工具介紹與選型指南

> 想做什麼？
>
> * **傳統 NLP 前處理**（分詞、POS/NER 前一步）
> * **LLM 訓練/推理**（子詞分解、token 計數、對齊）
> * **中文/日文/韓文** 等無空白語言的**斷詞**
> * **多語/跨領域**（學術、社群、程式碼）

---

## 一、核心類型速覽

| 類型                                  | 代表工具                                     | 主要用途               | 特點                       |
| ----------------------------------- | ---------------------------------------- | ------------------ | ------------------------ |
| **規則/詞典式斷詞**                        | Jieba、pkuseg、THULAC、MeCab、Sudachi、KoNLPy | 中/日/韓斷詞            | 安裝簡單、可自定辭典；對新詞與專有名詞需調詞庫  |
| **子詞模型（BPE / Unigram / WordPiece）** | SentencePiece、HF Tokenizers、tiktoken     | LLM 訓練與推理          | 語言不可知、可處理 OOV；與模型/檢索一致性高 |
| **通用 NLP 套件**                       | spaCy、NLTK、Stanza                        | 英語及多語 tokenization | 一站式 pipeline（POS/NER/依存） |
| **機器翻譯傳統工具**                        | Moses / Sacremoses                       | MT 資料前處理           | 歷史悠久、搭配 BLEU/評測常見        |
| **清洗/標準化**                          | ftfy、regex、Sacremoses                    | 清洗 + 正規化           | 適合雜訊文本（社群/爬蟲）            |

---

## 二、子詞模型（LLM 首選）

### 1) SentencePiece（Google）

* **演算法**：BPE、Unigram；**語言不可知**，不需預先斷詞。
* **常用於**：T5/mT5、ByT5、Whisper、LLaMA 系列等。
* **優點**：單一 `*.model`/`*.vocab`；訓練/使用一致。

**命令行（Unigram 訓練）**

```bash
spm_train --input=data.txt --model_prefix=spm_unigram --vocab_size=32000 --model_type=unigram
```

**Python**

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="spm_unigram.model")
ids = sp.encode("你好，LLM！", out_type=int)
text = sp.decode(ids)
```

### 2) Hugging Face Tokenizers（Rust/高速）

* **特色**：BPE / WordPiece / Unigram / ByteLevel；與 Transformers 無縫；**fast** 實作。
* **適合**：自訓 tokenizer、批量快速編碼、交付 `tokenizer.json`。

**Python（BPE）**

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=32000, min_frequency=2)
tok.train(files=["data.txt"], trainer=trainer)
tok.save("bpe_tokenizer.json")
```

### 3) tiktoken（OpenAI）

* **用途**：與 GPT 家族相容的 **token 計數 / 分詞**（計費、截斷、對齊）。

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("Count my tokens, please.")
len(ids)  # token 數
```

---

## 三、中文 / 日文 / 韓文斷詞

### 中文

* **Jieba**：簡單易用，支援自定詞庫。

```python
import jieba
list(jieba.cut("我愛自然語言處理與大型語言模型"))
```

* **pkuseg**：多領域模型（新聞/網絡/醫學）。
* **THULAC**：分詞 + 詞性標註。

### 日文

* **MeCab**（辭典：IPADic / UniDic）、**Sudachi/SudachiPy**

```bash
mecab -Owakati <<< "寿司が食べたい"
```

* Sudachi 有 **A/B/C** 三種分割模式（粗 → 細）。

### 韓文

* **KoNLPy**：封裝 Hannanum、Kkma、Mecab-Ko 等多種斷詞器。

> 子詞模型（SentencePiece/BPE）也能處理 CJK；但若需要**詞級標註/可讀詞彙**（例如 NER/關鍵詞），常見流程是：**先詞級斷詞 → 再對齊子詞**。

---

## 四、英語/通用 NLP 套件

### spaCy

* **優點**：高品質 tokenizer + POS/NER/依存；工業級。

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
[t.text for t in doc]  # token list
```

### NLTK / Stanza

* **NLTK**：教學研究常用，多種 tokenizer（正則、Punkt）。
* **Stanza（StanfordNLP）**：多語支持，PyTorch 後端。

---

## 五、機器翻譯與評測周邊

* **Moses / Sacremoses**：傳統 MT tokenization/正規化；
* **SacreBLEU**：固定參數與標準化，確保可重現的 BLEU/chrF 分數。

---

## 六、選型建議（10 秒版）

* **LLM 訓練/推理/計費/截斷** → SentencePiece / HF Tokenizers / tiktoken
* **CJK 需要詞級標註** → Jieba / pkuseg / THULAC / MeCab / Sudachi / KoNLPy
* **英語/多語管線（含 POS/NER/依存）** → spaCy / Stanza
* **對齊舊式 MT 流程/評測** → Moses/Sacremoses + SacreBLEU

---

## 七、常見坑位與實務技巧

1. **字元正規化**：全形/半形、Unicode 合字、雜訊文本（用 `ftfy`/正規化先處理）。
2. **空白與標點策略**：子詞常保留空白/特殊字元（ByteLevel）以利還原。
3. **詞彙表大小**：8k–16k（單語/小模型）、32k（常見多語/通用）、50k+（極多語/含程式碼）。
4. **對齊**：詞級標註 ↔ 子詞需 **offset mapping**（HF `encode_plus` 提供）。
5. **一致性**：**訓練/微調/推理** 必須用**同一 tokenizer**。
6. **效能**：大批量請用 **HF fast**（Rust）或批量 API；I/O 可能是瓶頸。

---

## 八、快速對照

**SentencePiece（通用子詞）**

```bash
spm_train --input=data.txt --model_prefix=spm --vocab_size=32000 --model_type=bpe
```

**HF Tokenizers（高速自訓）**

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("bpe_tokenizer.json")
ids = tok.encode("Hello 世界").ids
```

**Jieba（中文分詞）**

```python
import jieba
list(jieba.cut("深度學習的分詞與子詞模型"))
```

---

## 九、進階：自訓 tokenizer 演算法怎麼選？

* **BPE**：穩健、廣泛、可控；字節級 BPE 對噪聲更耐受。
* **Unigram（SentencePiece 預設）**：機率模型選子詞集合，常更平滑。
* **WordPiece**：BERT 經典，與 Google 生態相容。
* **ByteLevel**：避免 Unicode 地雷，跨語種友好（表情符號/控制字元）。





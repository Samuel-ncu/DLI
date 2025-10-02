
# GPTQ（Gradient Post Training Quantization） 與 QAT （Quantization-Aware Training）

## 1. 量化 (Quantization) 概念

量化是將高精度浮點 (FP32) 權重與激活值壓縮為低精度 (如 INT8、FP8、4bit)，以 **減少模型大小、加速推論、降低功耗**。

| 方法 | 訓練需求 | 精度 | 優點 | 缺點 |
|------|---------|------|------|------|
| **PTQ** (Post Training Quantization) | 不需再訓練 | 中等 | 快速、簡單 | 精度掉落較大 |
| **GPTQ** (Gradient Post Training Quantization) | 不需再訓練，但用梯度調整 | 中高 | 精度較 PTQ 好，快速 | 稍複雜 |
| **QAT** (Quantization-Aware Training) | 需再訓練 | 高 | 精度接近 FP32 | 訓練成本高 |

---

## 2. GPTQ（Gradient Post Training Quantization）

### 原理
- 使用 **Hessian 近似** 與 **梯度資訊** 來最小化量化誤差。
- 不需完整資料，只需少量校準資料即可快速完成高品質量化。

**數學表示：**

$$
\min_Q \| W - Q(W) \|_H^2 = (W - \hat{W})^T H (W - \hat{W})
$$

- \(W\)：原始權重矩陣  
- \(\hat{W}\)：量化後權重  
- \(H\)：Hessian 近似，用於評估權重對輸出誤差的敏感度

---

### 🔍 Hessian 近似是什麼

**Hessian** 是損失函數 \(L(\mathbf{w})\) 對權重 \(\mathbf{w}\) 的二階導數矩陣：

$$
H = \nabla^2_{\mathbf{w}} L(\mathbf{w})
$$

- 一階導數 (gradient) 告訴你下降方向。
- 二階導數 (Hessian) 告訴你每個權重的重要性與曲率。

完整 Hessian 計算量極大，因此 GPTQ 使用 **近似**：
1. **對角近似**：僅取對角線元素。
2. **樣本估計**：用少量校準資料計算輸入特徵協方差。
3. **分組近似**：分成小區塊計算，降低記憶體成本。

實務上：

$$
H \approx \frac{1}{N} X^T X
$$

其中 \(X\) 為校準資料在該層的輸入特徵。

> ✅ GPTQ 用這個近似 Hessian 來判斷哪些權重最重要，先保護重要權重，減少量化誤差。

---

### 實作範例

```bash
pip install auto-gptq
````

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_name = "facebook/opt-1.3b"

quantize_config = BaseQuantizeConfig(
    bits=4,           # 支援 4 或 8 bit
    group_size=128,   # 分組大小
    damp_percent=0.01 # Hessian 阻尼係數
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用少量校準資料
examples = ["Hello world!", "This is GPTQ test."]
model.quantize(examples)

model.save_quantized("opt-1.3b-gptq")
```

---

## 3. QAT（Quantization-Aware Training）

### 原理

* 在訓練時插入 **Fake Quantization Node**，模擬量化誤差。
* 前向路徑進行量化近似，反向則使用 **Straight-Through Estimator (STE)**，讓梯度通過。

#### 前向傳遞 (Forward)

模擬量化：

$$
\hat{x} = \text{clamp}\big(\text{round}(x / s), q_{\min}, q_{\max}\big) \cdot s
$$

* (s)：量化比例（scale factor）
* (q_{\min}, q_{\max})：整數範圍（例如 INT8 為 ([-128,127])）

#### 反向傳遞 (Backward)

`round()` 不可微，QAT 用 **STE** 近似：

$$
\frac{\partial \hat{x}}{\partial x} \approx 1
$$

也就是在反向傳遞時，假裝量化函數是恆等函數 (f(x)=x)，讓梯度可以直接傳回去。

---

### STE 在哪裡被使用

在 QAT 中，每個 **Fake Quantization Node**（假量化節點）都會用到 STE。

* **權重量化 (Weight FakeQuant)**：在 Conv/Linear 層的權重上加入 fake quant。
* **激活量化 (Activation FakeQuant)**：在層輸出或 ReLU 後加入 fake quant。

**流程示意：**

```text
[Conv / Linear 層權重] → FakeQuant(round) → 前向傳遞結果
                                 │
                                 └─ 反向傳遞時用 STE (梯度 ≈ 1)
```

在 PyTorch 中，`FakeQuantize` 模組的原理如下：

```python
class FakeQuantize(nn.Module):
    def forward(self, X):
        # 前向：模擬量化
        Xq = torch.round(X / self.scale) * self.scale
        return Xq

    def backward(self, grad_output):
        # 反向：STE
        return grad_output  # 假裝梯度不受 round() 影響
```

---

### 實作範例 (PyTorch)

```bash
pip install torch torchvision
```

```python
import torch
import torch.nn as nn
import torch.quantization as quant

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = Net()

# 準備 QAT
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(5):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()

# 轉換成量化模型
quant.convert(model.eval(), inplace=True)
torch.save(model.state_dict(), "qat_model_int8.pth")
```

---

## 4. GPTQ 與 QAT 比較表

| 特性      | GPTQ                   | QAT                         |
| ------- | ---------------------- | --------------------------- |
| 是否需要再訓練 | ❌                      | ✅                           |
| 所需資料    | 少量校準資料                 | 全部訓練資料                      |
| 精度      | 高 (略低於 QAT)            | 最佳                          |
| 計算成本    | 低                      | 高                           |
| 適用場景    | 已無法再訓練的大型模型            | 可重新訓練或微調                    |
| 工具支援    | AutoGPTQ, bitsandbytes | PyTorch QAT, TensorFlow QAT |

---

## 5. 其他常見量化技術

| 技術              | 說明                                   | 適用場景         |
| --------------- | ------------------------------------ | ------------ |
| **SmoothQuant** | 先平滑化權重與 activation，降低量化誤差            | LLM PTQ 前處理  |
| **AWQ**         | Activation-aware Weight Quantization | 針對激活範圍優化權重量化 |
| **ZeroQuant**   | 用混合精度快速量化                            | 大型語言模型       |
| **FP8 QAT**     | NVIDIA Hopper 架構支援 FP8               | 高效能 GPU      |

---

## 6. 部署選項

| 部署技術                    | 說明                                           |
| ----------------------- | -------------------------------------------- |
| **ONNX Runtime + INT8** | 跨平台部署                                        |
| **TensorRT**            | NVIDIA GPU 上最佳化，支援 FP16 / INT8，對 LLM 有極高加速效果 |
| **vLLM + GPTQ**         | LLM 4bit 高效推論                                |

### 🚀 TensorRT 詳細說明

**TensorRT** 是 NVIDIA 推出的高效推論引擎，專門用於 GPU 加速。

* **優點：**

  * 支援 FP16、INT8、FP8（最新版本）
  * 自動圖優化 (layer fusion、kernel auto-tuning)
  * 高效記憶體管理與動態 batch size
  * 對 LLM、CNN、Transformer 類模型都有極佳加速效果

* **典型工作流程：**

  1. 將模型匯出成 ONNX 格式
  2. 用 `trtexec` 或 Python API 轉換成 TensorRT Engine
  3. 部署 Engine 檔執行推論

```bash
# 轉換 ONNX 到 TensorRT INT8
trtexec --onnx=model.onnx --int8 --saveEngine=model_int8.engine --workspace=4096
```

```python
# Python API 使用 TensorRT
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

builder.max_batch_size = 8
builder.int8_mode = True
engine = builder.build_cuda_engine(network)

with open("model_int8.engine", "wb") as f:
    f.write(engine.serialize())
```

* **建議：**

  * 如果模型經過 QAT，INT8 精度幾乎與 FP32 接近。
  * 若用 GPTQ 4bit，目前 TensorRT 仍以 INT8/FP16 為主，但可先用 GPTQ 壓縮，再透過 ONNX/TensorRT 加速。

---

## 7. 學習建議路線

1. **了解量化理論**（Uniform / Non-uniform, Per-channel）
2. **玩 GPTQ**：用 `auto-gptq` 對 HuggingFace 模型量化
3. **進階 QAT**：使用小模型練習，再應用到實務專案
4. **部署優化**：學習 TensorRT、ONNX Runtime、vLLM

---

## ✅ 結論

* **GPTQ**：快速、免重訓，適合已無法重訓的 LLM。
* **QAT**：需要再訓練，但精度最高。
* **STE**：是 QAT 成功的關鍵，允許梯度穿過不可微的量化函數。
* **Hessian 近似**：是 GPTQ 的關鍵，能判斷哪些權重最重要，優化量化後的精度。
* **TensorRT**：最佳化 GPU 推論，結合 QAT/ONNX 可達到工業級效能與穩定精度。
* **建議策略**：

  * 想快速壓縮 LLM → **GPTQ + vLLM 或 TensorRT**
  * 工業級高精度 → **QAT + TensorRT**
  * 可先用 SmoothQuant / AWQ 做 PTQ，再用少量 QAT fine-tune。



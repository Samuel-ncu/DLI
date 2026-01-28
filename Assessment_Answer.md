# Setup
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

model_path = "http://0.0.0.0:9004/v1"
%env NVIDIA_BASE_URL=$model_path
%env NVIDIA_DEFAULT_MODE=open

model_name = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"


## Maybe you'll need more connectors? Maybe you'll need a different endpoint (this one may be down)? 
llm = ChatNVIDIA(
    model=model_name, 
    base_url=model_path, 
    max_completion_tokens=5000, 
    temperature=0
)
```

# [Task 1] Image Ingestion
```python
import base64
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage

# ==========================================
# 1. 環境設定與模型初始化
# ==========================================
# 請注意：這裡我們換成了具備視覺能力的模型 ID
model_name = "meta/llama-3.2-11b-vision-instruct" 
model_path = "http://0.0.0.0:9004/v1"

# 設定環境變數 (LangChain 連接器會讀取這些)
%env NVIDIA_BASE_URL=$model_path
%env NVIDIA_DEFAULT_MODE=open

# 初始化 ChatNVIDIA
llm = ChatNVIDIA(
    model=model_name, 
    base_url=model_path, 
    max_completion_tokens=5000, 
    temperature=0
)

# ==========================================
# 2. 實作 ask_about_image 方法
# ==========================================
def ask_about_image(image_path: str, question: str = "Describe the image") -> str:
    """
    使用 ChatNVIDIA 處理圖片與問題。
    """
    # --- A. 讀取圖片並進行 Base64 編碼 ---
    if not os.path.exists(image_path):
        return f"錯誤：找不到圖片 {image_path}"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # 這裡假設為 PNG，若要更嚴謹可動態判斷副檔名
    data_url = f"data:image/png;base64,{image_b64}"

    # --- B. 建立多模態訊息 ---
    # LangChain 的格式是將文字與圖片網址放在內容清單中
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    # --- C. 呼叫模型 ---
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"模型呼叫失敗: {str(e)}"

# ==========================================
# 3. 執行與測試
# ==========================================
description = ask_about_image("./imgs/agent-overview.png", "Describe the image")
print(description)
```

# [Task 2] Image Creation
```python
from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

# ==========================================
# 1. 初始化 Pipeline (放在函數外，避免重複載入)
# ==========================================
# 建議先清理一下 GPU 記憶體，確保有足夠空間給 SDXL
if torch.cuda.is_available():
    torch.cuda.empty_cache()

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

# ==========================================
# 2. 定義輔助函數：繪製圖片
# ==========================================
def plot_imgs(images, rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    if rows * cols == 1:
        axes.imshow(images[0])
        axes.axis('off')
    else:
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. 實作 generate_images 方法
# ==========================================
def generate_images(prompt: str):
    """
    根據文字描述生成圖片
    """
    ####################################################################
    ## < EXERCISE SCOPE
    
    # 執行生成過程
    # num_inference_steps 設定為 30-50 效果通常較好，20-25 速度較快
    output = pipe(
        prompt=prompt, 
        num_inference_steps=25, 
        guidance_scale=7.5
    )
    
    return output.images 
    
    ## EXERCISE SCOPE >
    ####################################################################

# ==========================================
# 4. 執行生成
# ==========================================
description = ("The image presents a diagram illustrating the relationship between an agent and "
               "various components of memory and tools. The diagram is structured with a central "
               "rectangle labeled 'Agent,' which is connected to several other elements...")

print("正在根據描述生成圖片，請稍候...")
generated_images = generate_images(description)

# 顯示圖片
for img in generated_images:
    # 如果在環境中支援，可以使用 img.show()
    # 在 Notebook 中，我們通常使用 plot_imgs
    pass

plot_imgs(generated_images, 1, 1)
```

# [Task 3] Prompt Synthesis
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def llm_rewrite_to_image_prompts(user_query: str, n: int = 4) -> list[str]:
    ####################################################################
    ## < EXERCISE SCOPE
    
    # 1. 定義 Prompt Template
    # 我們告訴 LLM 它是一個專業的 Stable Diffusion 提示詞工程師
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional prompt engineer for Stable Diffusion XL. "
            "Your task is to take a complex technical description of an AI architecture "
            "and rewrite it into {n} distinct, highly visual, and concise image generation prompts. "
            "Focus on style, lighting, and layout. Avoid technical jargon; use visual metaphors. "
            "Output each prompt on a new line, starting with a dash (-)."
        )),
        ("human", "Convert this description into {n} visual prompts: {user_query}")
    ])

    # 2. 建立 Chain
    # 使用之前定義好的 llm (Llama-3.3-70B 非常適合這個任務)
    chain = prompt_template | llm | StrOutputParser()

    # 3. 執行並解析輸出
    raw_output = chain.invoke({"user_query": user_query, "n": n})
    
    # 將輸出的字串按行切割，並清理多餘的符號
    # 我們預期輸出是像 "- Prompt 1 \n - Prompt 2" 這樣的格式
    lines = [line.strip("- ").strip() for line in raw_output.strip().split("\n") if line.strip()]
    
    # 確保返回的數量正確 (截斷或填充)
    sd_prompts = lines[:n]
    
    # 防禦性程式碼：如果 LLM 回傳不足 n 個，重複最後一個
    while len(sd_prompts) < n:
        sd_prompts.append(sd_prompts[-1] if sd_prompts else "A clean technical diagram of an AI agent")

    assert len(sd_prompts) == n
    return sd_prompts
    
    ## EXERCISE SCOPE >
    ####################################################################

# --- 執行合成 ---
# 使用先前 Task 1 得到的 description
new_sd_prompts = llm_rewrite_to_image_prompts(description, n=4)

print("🚀 生成的新提示詞：")
for i, p in enumerate(new_sd_prompts):
    print(f"{i+1}. {p}")
```

# [Task 4] Pipelining and Iterating



```python
def generate_images_from_image(image_url: str, num_images = 4):
    print(f"\n--- 開始處理: {image_url} ---")

    ####################################################################
    ## < EXERCISE SCOPE

    # 1. 描述圖片：使用 Task 1 的視覺模型獲取原始描述
    # 注意：這裡確保使用的是支援視覺的模型（如 Nemotron-VL 或 Llama-3.2-Vision）
    original_description = ask_about_image(image_url, "Provide a highly detailed technical and visual description of this image.")
    print(f"✅ 已生成原始描述 (長度: {len(original_description)})")

    # 2. 提示詞合成：將冗長的描述轉換為 n 個適合 Diffusion 的短提示詞
    # 我們使用 Task 3 實作的 LLM 重寫邏輯
    diffusion_prompts = llm_rewrite_to_image_prompts(original_description, n=num_images)
    print(f"✅ 已合成 {num_images} 組合成提示詞")

    # 3. 影像生成：迭代提示詞並產出影像
    generated_images = []
    for i, prompt in enumerate(diffusion_prompts):
        print(f"🎨 正在生成第 {i+1}/{num_images} 張圖片...")
        # 呼叫 Task 2 的 Diffusion Pipeline
        # 為了速度，我們可以稍微降低 num_inference_steps
        imgs = generate_images(prompt) 
        generated_images.extend(imgs)

    ## EXERCISE SCOPE >
    ####################################################################

    # 視覺化展示結果
    print(f"✨ 正在展示 {image_url} 的轉換成果：")
    plot_imgs(generated_images, rows=1, cols=num_images)
    
    return generated_images, diffusion_prompts, original_description

# --- 批次執行評測 ---
all_results = []
test_files = [
    "imgs/agent-overview.png",
    "imgs/multimodal.png",
    "img-files/tree-frog.jpg",
    "img-files/paint-cat.jpg"
]

for file_path in test_files:
    if os.path.exists(file_path):
        res = generate_images_from_image(file_path)
        all_results.append(res)
    else:
        print(f"⚠️ 略過：找不到檔案 {file_path}")
```



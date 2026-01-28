# Setup
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

model_path = "http://0.0.0.0:9004/v1"
%env NVIDIA_BASE_URL=$model_path
%env NVIDIA_DEFAULT_MODE=open

model_name = "meta/llama-3.2-11b-vision-instruct"

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
import requests
import base64

def ask_about_image(image_path: str, question: str = "Describe the image") -> str:
    ####################################################################
    ## < EXERCISE SCOPE

    ## 1. 將圖片讀取並轉換為 Base64 字串
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    ## 2. 使用之前定義好的 llm (ChatNVIDIA) 實例發送請求
    ## 注意：多模態模型通常要求輸入特定的訊息格式
    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    )

    response = llm.invoke([message])
    return response.content

    ## EXERCISE SCOPE >
    ####################################################################

description = ask_about_image("imgs/agent-overview.png", "Describe the image")
print(description)
```

# [Task 2] Image Creation
```python
from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

## TODO: Implement this method as appropriate
def generate_images(prompt: str, n: int = 1):
    ####################################################################
    ## < EXERCISE SCOPE

    ## 1. 載入預訓練的擴散模型（例如 SDXL Turbo 或 SD 1.5）
    ## 註：在 NVIDIA 課程環境中，通常會預載模型。這裡使用一個常見的快取路徑或模型名稱。
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

    ## 2. 生成圖片
    # 這裡直接生成 n 張圖片
    gen_images = pipe(prompt, num_images_per_prompt=n).images

    ## 3. 將圖片儲存為檔案並回傳路徑清單（以便 plot_imgs 讀取）
    image_paths = []
    for i, img in enumerate(gen_images):
        path = f"generated_img_{i}.png"
        img.save(path)
        image_paths.append(path)

    return image_paths

    ## EXERCISE SCOPE >
    ####################################################################

def plot_imgs(image_paths, r=2, c=2):
    fig, axes = plt.subplots(r, c)
    for i, ax in enumerate(getattr(axes, "flat", [axes])):
        if i < len(image_paths):
            img = plt.imread(image_paths[i])
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 執行 Task 2
images = generate_images(description)
plot_imgs(images, 1, 1)
```

# [Task 3] Prompt Synthesis
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def llm_rewrite_to_image_prompts(user_query: str, n: int = 4) -> list[str]:

    ####################################################################
    ## < EXERCISE SCOPE
    
    ## 1. 定義 Prompt Template，要求模型將敘述轉換成 n 個適合 SD 的短提示詞
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert prompt engineer. Rewrite the description into {n} distinct, concise, high-quality prompts for Stable Diffusion. Return ONLY the prompts, one per line, no numbering."),
        ("human", "{user_query}")
    ])

    ## 2. 建立簡單的 Chain 並執行
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"user_query": user_query, "n": n})

    ## 3. 將字串解析成列表並確保長度為 n
    sd_prompts = [p.strip() for p in response.split('\n') if p.strip()][:n]
    
    # 防呆處理：若數量不足則重複最後一項或補位
    while len(sd_prompts) < n:
        sd_prompts.append(sd_prompts[-1] if sd_prompts else "highly detailed art")

    assert len(sd_prompts) == n
    return sd_prompts
    
    ## EXERCISE SCOPE >
    ####################################################################

# 測試轉換邏輯
new_sd_prompts = llm_rewrite_to_image_prompts(description, n=4)
print(new_sd_prompts)
```

# [Task 4] Pipelining and Iterating



```python
## TODO: Execute on assessment objective
def generate_images_from_image(image_url: str, num_images = 4):

    print(f"Generating images for {image_url}")

    ####################################################################
    ## < EXERCISE SCOPE

    ## 1. 產生原始圖片的描述 (呼叫 Task 1 實作的函式)
    original_description = ask_about_image(image_url, "Describe this image in detail.")

    ## 2. 將描述轉換為 4 個不同的 Diffusion Prompts (呼叫 Task 3 實作的函式)
    diffusion_prompts = llm_rewrite_to_image_prompts(original_description, n=num_images)

    ## 3. 根據生成的 Prompts 產生圖片 (使用 Task 2 的邏輯)
    # 為了效能，我們在這裡載入一次 Pipeline
    from diffusers import DiffusionPipeline
    import torch
    
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    
    generated_images = []
    for i, prompt in enumerate(diffusion_prompts):
        # 生成圖片
        image = pipe(prompt, num_images_per_prompt=1).images[0]
        # 儲存圖片並給予唯一檔名
        path = f"output_{i}_{image_url.split('/')[-1]}"
        image.save(path)
        generated_images.append(path)

    ## EXERCISE SCOPE >
    ####################################################################

    plot_imgs(generated_images)
    return generated_images, diffusion_prompts, original_description

results = []
results += [generate_images_from_image("imgs/agent-overview.png")]
results += [generate_images_from_image("imgs/multimodal.png")]
results += [generate_images_from_image("img-files/tree-frog.jpg")]
results += [generate_images_from_image("img-files/paint-cat.jpg")]
```



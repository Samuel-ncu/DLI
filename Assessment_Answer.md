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
import requests
import base64
import os

# ==========================================
# 1. 設定模型資訊 (根據您的清單更新)
# ==========================================
# 這裡使用支援視覺 (VL) 的模型，並指向正確的 Port (9002)
MODEL_PATH = "http://0.0.0.0:9002/v1" 
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"

def ask_about_image(image_path: str, question: str = "Describe the image") -> str:
    """
    將圖片編碼為 Base64 並傳送至 VLM 模型進行分析。
    """
    # --- A. 檢查檔案 ---
    if not os.path.exists(image_path):
        return f"錯誤：找不到圖片檔案 {image_path}"

    # --- B. 判斷 MIME 類型並進行 Base64 編碼 ---
    ext = image_path.lower().split(".")[-1]
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp"
    }
    mime = mime_map.get(ext, "image/png")

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        # 格式化為 Data URL
        data_url = f"data:{mime};base64,{image_b64}"
    except Exception as e:
        return f"圖片讀取/編碼失敗: {str(e)}"

    # --- C. 準備 Payload (OpenAI 兼容格式) ---
    url = f"{MODEL_PATH.rstrip('/')}/chat/completions"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    # 在本地端 vLLM 環境，通常不需要 Authorization，或留空即可
    headers = {"Content-Type": "application/json"}

    # --- D. 發送請求與處理結果 ---
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        
        # 如果失敗，印出詳細原因方便除錯
        if resp.status_code != 200:
            error_detail = resp.text
            return f"伺服器錯誤 (Status {resp.status_code}): {error_detail}"

        result = resp.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"網路連線異常: {str(e)}"

# ==========================================
# 2. 執行測試
# ==========================================
if __name__ == "__main__":
    # 請確保此路徑下確實有這張圖片
    test_image = "./imgs/agent-overview.png"
    test_question = "Describe the image"

    print(f"正在分析圖片：{test_image} ...")
    description = ask_about_image(test_image, test_question)
    
    print("\n--- 模型分析結果 ---")
    print(description)
```

# [Task 2] Image Creation
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")


## TODO: Consider initializing your diffusion pipeline outside of generate_images

## TODO: Implement this method
def generate_images(prompt):
    ####################################################################
    ## < EXERCISE SCOPE
    
    images = pipe(prompt=prompt).images

    return images 
    ## EXERCISE SCOPE >
    ####################################################################
description = "The image presents a diagram illustrating the relationship between an agent and various components of memory and tools. The diagram is structured with a central rectangle labeled 'Agent,' which is connected to several other elements. On the left side, there are three rectangles representing 'Short-term memory,' 'Memory,' and 'Tools,' each with a list of items beneath them. 'Short-term memory' includes 'Calendar ()', 'Calculator ()', and 'CodeInterpreter ()', while 'Memory' has 'Long-term memory' and '...more' listed. 'Tools' is connected to 'Action' and 'Planning.' On the right side, there are three rectangles labeled 'Reflection,' 'Self-critics,' and 'Chain of thoughts,' which are connected to 'Planning.' Additionally, there is a rectangle labeled 'Subgoal decomposition' connected to 'Action.' The connections between the elements are depicted with solid and dashed lines, indicating different types of relationships."


    ## EXERCISE SCOPE >
    ####################################################################

images = generate_images(description)
for img in images:
    img.show()
```

# [Task 3] Prompt Synthesis
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt

####################################################################
## < EXERCISE SCOPE

# 1) Prompt template: description -> ONE diffusion prompt (one line)
prompt_tmpl = ChatPromptTemplate.from_messages([
    ("system",
     "You are a prompt engineer for text-to-image diffusion models (Stable Diffusion / SDXL). "
     "Convert the user's description into ONE high-quality, keyword-rich diffusion prompt. "
     "Return ONE LINE ONLY (no bullets, no numbering, no quotes, no explanations)."),
    ("user",
     "User description:\n{desc}\n\n"
     "Write a diffusion prompt including: subject, setting, composition, lighting, camera/lens, style, and quality tags. "
     "Output only the prompt line.")
])

# 2) Chain: template -> llm -> string
diff_prompt_chain = prompt_tmpl | llm | StrOutputParser()

# 3) Emphasis variants (to create multiple prompts)
emphases = [
    "cinematic wide shot, environment detail, soft volumetric lighting",
    "close-up, shallow depth of field, bokeh, portrait composition",
    "top-down / isometric composition, clean shapes, high clarity",
    "dramatic angle, high contrast lighting, stylized artistic look",
]

# 4) Build 4 synthetic prompts from an image description string `description`
new_sd_prompts = []
for e in emphases:
    p = diff_prompt_chain.invoke({"desc": f"{description}\n\nEmphasis: {e}"}).strip()
    new_sd_prompts.append(p)

print("len(new_sd_prompts) =", len(new_sd_prompts))
for i, p in enumerate(new_sd_prompts, 1):
    print(f"{i}. {p}")

# 5) (Optional) single prompt version
new_diff_prompt = diff_prompt_chain.invoke({"desc": description}).strip()

## EXERCISE SCOPE >
####################################################################


# Utility: display images returned as file paths OR PIL images
def plot_imgs(image_paths_or_pils, r=2, c=2):
    fig, axes = plt.subplots(r, c, figsize=(4*c, 4*r))
    axes_list = getattr(axes, "flat", [axes])

    for i, ax in enumerate(axes_list):
        ax.axis("off")
        if i >= len(image_paths_or_pils):
            continue

        item = image_paths_or_pils[i]

        # Case A: filepath
        if isinstance(item, str):
            img = plt.imread(item)
            ax.imshow(img)

        # Case B: PIL image
        elif hasattr(item, "size") and hasattr(item, "mode"):
            ax.imshow(item)

        else:
            ax.set_title("Unsupported type")

    plt.tight_layout()
    plt.show()

```

# [Task 4] Pipelining and Iterating



```python
import os
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def llm_rewrite_to_image_prompts(user_query: str, n: int = 4) -> list[str]:
    ####################################################################
    ## < EXERCISE SCOPE

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You write prompts for text-to-image diffusion models (Stable Diffusion / SDXL). "
         "Output EXACTLY N lines. Each line is ONE prompt. No numbering/bullets/quotes/explanations."),
        ("user",
         "Image description:\n{desc}\n\n"
         "Generate exactly {n} distinct diffusion prompts. "
         "Each line should include: subject, setting, composition, lighting, camera/lens, style, and quality tags. "
         "Output ONLY the {n} lines.")
    ])

    chain = prompt | llm | StrOutputParser()

    text = ""
    for _ in range(2):  # retry twice to reduce backend timeout impact
        try:
            text = chain.invoke({"desc": user_query, "n": n})
            if text:
                break
        except Exception:
            text = ""

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    # normalize if model adds bullets/numbering
    cleaned = []
    for ln in lines:
        while ln and ln[0] in "-•":
            ln = ln[1:].strip()
        if len(ln) >= 3 and ln[0].isdigit() and ln[1] in ".)" and ln[2] == " ":
            ln = ln[3:].strip()
        cleaned.append(ln)

    sd_prompts = cleaned[:n]

    # fallback prompts if LLM failed/timeout/empty
    if len(sd_prompts) < n:
        base = (user_query or "").strip().replace("\n", " ")
        if not base:
            base = "a high quality photo of an interesting subject"
        fallback = [
            f"{base}, cinematic wide shot, environment detail, soft volumetric lighting, ultra detailed, sharp focus, high quality",
            f"{base}, close-up, shallow depth of field, bokeh, portrait composition, ultra detailed, sharp focus, high quality",
            f"{base}, top-down composition, clean shapes, high clarity, studio lighting, ultra detailed, sharp focus, high quality",
            f"{base}, dramatic angle, high contrast lighting, stylized artistic look, ultra detailed, sharp focus, high quality",
        ]
        # append until length n
        for f in fallback:
            if len(sd_prompts) >= n:
                break
            sd_prompts.append(f)

    while len(sd_prompts) < n:
        sd_prompts.append(sd_prompts[-1])

    assert len(sd_prompts) == n
    return sd_prompts

    ## EXERCISE SCOPE >
    ####################################################################


def _show_paths_grid(image_paths: list[str], r=2, c=2, title=None):
    """Display images from file paths (best-effort)."""
    if not image_paths:
        print("No images to display.")
        return
    n = len(image_paths)
    fig, axes = plt.subplots(r, c, figsize=(4*c, 4*r))
    axes_list = getattr(axes, "flat", [axes])
    for i, ax in enumerate(axes_list):
        ax.axis("off")
        if i < n:
            try:
                img = plt.imread(image_paths[i])
                ax.imshow(img)
                ax.set_title(f"{i+1}", fontsize=10)
            except Exception:
                ax.set_title(f"Failed: {image_paths[i]}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


## TODO: Execute on assessment objective
def generate_images_from_image(image_url: str, num_images = 4):

    ####################################################################
    ## < EXERCISE SCOPE

    ## TODO 1: Generate the description of the image provided in image_url
    original_description = ask_about_image(image_url, "Describe the image in detail")

    ## TODO 2: Generate four disjoint prompts, hopefully different, to feed into SDXL
    diffusion_prompts = llm_rewrite_to_image_prompts(original_description, n=num_images)

    ## TODO 3: Generate the resulting images
    # IMPORTANT: must return PATH STRINGS so send_metadata() can do .replace()
    # Save locally as needed.
    save_dir = "/dli/task/generated_images" if os.path.exists("/dli/task") else "generated_images"
    os.makedirs(save_dir, exist_ok=True)

    generated_images = []  # list[str] paths

    for i, p in enumerate(diffusion_prompts):
        outs = []
        for _ in range(2):  # retry per prompt
            try:
                outs = generate_images(p)  # -> [filepath] OR [PIL Image]
                if outs:
                    break
            except Exception:
                outs = []

        if not outs:
            continue

        first = outs[0]

        # Case A: already a filepath
        if isinstance(first, str):
            generated_images.append(first)
            continue

        # Case B: PIL image -> save to path
        if hasattr(first, "save") and hasattr(first, "size"):
            out_path = os.path.join(save_dir, f"task4_{os.path.basename(image_url).replace('.','_')}_{i}.png")
            try:
                first.save(out_path)
                generated_images.append(out_path)
            except Exception:
                pass

    # Ensure we have up to num_images paths (best-effort; do not crash)
    if len(generated_images) > num_images:
        generated_images = generated_images[:num_images]

    # Display images (2x2 grid recommended)
    _show_paths_grid(generated_images, r=2, c=2, title=os.path.basename(image_url))

    ## EXERCISE SCOPE >
    ####################################################################
        
    return generated_images, diffusion_prompts, original_description


# Run
results = []
results += [generate_images_from_image("imgs/agent-overview.png")]
results += [generate_images_from_image("imgs/multimodal.png")]
results += [generate_images_from_image("img-files/tree-frog.jpg")]
results += [generate_images_from_image("img-files/paint-cat.jpg")]

```



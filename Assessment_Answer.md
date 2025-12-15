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

def ask_about_image(image_path: str, question: str = "Describe the image") -> str:
    ####################################################################
    ## < EXERCISE SCOPE

    # 1) pick mime type without extra imports
    ext = image_path.lower().split(".")[-1]
    if ext == "png":
        mime = "image/png"
    elif ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/png"  # safe fallback

    # 2) read + base64 encode
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime};base64,{image_b64}"

    # 3) OpenAI-compatible vLLM endpoint
    url = f"{model_path.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 800,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer None",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    out = resp.json()

    # 4) extract text
    try:
        return out["choices"][0]["message"]["content"]
    except Exception:
        return str(out)

    ## EXERCISE SCOPE >
    ####################################################################

description = ask_about_image("./imgs/agent-overview.png", "Describe the image")
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



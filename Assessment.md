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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def llm_rewrite_to_image_prompts(user_query: str, n: int = 4) -> list[str]:
    ####################################################################
    ## < EXERCISE SCOPE

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a prompt engineer for text-to-image diffusion models (e.g., Stable Diffusion / SDXL). "
         "Rewrite the user's description into high-quality, keyword-rich diffusion prompts. "
         "Each prompt must be ONE LINE ONLY, no numbering, no quotes, no bullets, no extra commentary."),
        ("user",
         "User description:\n{user_query}\n\n"
         "Generate exactly {n} distinct diffusion prompts. "
         "Each line should include: subject, setting, composition, lighting, camera/lens, style, and quality tags. "
         "Avoid unsafe content. Output ONLY the {n} lines.")
    ])

    chain = prompt | llm | StrOutputParser()
    text = chain.invoke({"user_query": user_query, "n": n})

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # If the model accidentally returns numbered lines, strip common prefixes
    cleaned = []
    for ln in lines:
        # remove leading "1. ", "- ", etc. without regex/imports
        while ln and (ln[0] in "-•"):
            ln = ln[1:].strip()
        if len(ln) >= 3 and ln[0].isdigit() and ln[1] in ".)" and ln[2] == " ":
            ln = ln[3:].strip()
        cleaned.append(ln)

    sd_prompts = cleaned[:n]
    while len(sd_prompts) < n:
        sd_prompts.append(sd_prompts[-1] if sd_prompts else user_query)

    assert len(sd_prompts) == n
    return sd_prompts

    ## EXERCISE SCOPE >
    ####################################################################

new_sd_prompts = llm_rewrite_to_image_prompts(description, n=4)

print("type(new_sd_prompts) =", type(new_sd_prompts))
print("len(new_sd_prompts) =", len(new_sd_prompts))
print("new_sd_prompts =", new_sd_prompts)
```

# [Task 4] Pipelining and Iterating
```python
## TODO: Execute on assessment objective
def generate_images_from_image(image_url: str, num_images = 4):

    print(f"Generating images for {image_url}")

    ####################################################################
    ## < EXERCISE SCOPE

    # 1) VLM: image -> description
    # (uses your existing ask_about_image; make sure it points to a vision model id that exists)
    original_description = ask_about_image(image_url, "Describe the image in detail")

    # 2) LLM: description -> N synthetic diffusion prompts
    diffusion_prompts = llm_rewrite_to_image_prompts(original_description, n=num_images)

    # 3) Diffusion: prompts -> images (return PIL Images)
    generated_images = []      # PIL Images to return
    generated_paths = []       # optional: file paths for plotting if plot_imgs expects paths

    # generate_images(...) from Task 2 returns file paths in our earlier implementation.
    # We'll convert to PIL so this function returns PIL images as requested.
    try:
        from PIL import Image
    except Exception:
        Image = None

    for i, sd_prompt in enumerate(diffusion_prompts):
        outs = generate_images(sd_prompt, n=1)  # may return [path] or [PIL]

        if not outs:
            continue

        first = outs[0]
        if hasattr(first, "save") and hasattr(first, "size"):
            # looks like a PIL image already
            generated_images.append(first)
            # also save a copy for plotting if needed
            try:
                p = f"gen_task4_{i}.png"
                first.save(p)
                generated_paths.append(p)
            except Exception:
                pass
        elif isinstance(first, str):
            # it's a file path
            generated_paths.append(first)
            if Image is not None:
                generated_images.append(Image.open(first).convert("RGB"))
        else:
            # unknown type; best-effort ignore
            pass

    # 4) Display (works whether plot_imgs expects paths or PILs)
    try:
        plot_imgs(generated_images)  # if plot_imgs supports PIL
    except Exception:
        try:
            plot_imgs(generated_paths)  # if plot_imgs expects paths
        except Exception:
            pass

    return generated_images, diffusion_prompts, original_description

    ## EXERCISE SCOPE >
    ####################################################################

results = []
results += [generate_images_from_image("imgs/agent-overview.png")]
results += [generate_images_from_image("imgs/multimodal.png")]
results += [generate_images_from_image("img-files/tree-frog.jpg")]
results += [generate_images_from_image("img-files/paint-cat.jpg")]
```



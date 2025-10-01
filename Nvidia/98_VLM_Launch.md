<center><a href="https://www.nvidia.com/en-us/training/"><img src="https://dli-lms.s3.amazonaws.com/assets/general/DLI_Header_White.png" width="400" height="186" /></a></center>

# **Notebook 98:** Launching Your vLLM Servers

Please run the following command to kickstart a vLLM OpenAI-style server running a local Visual Language Model (in this case, [**Microsoft's Phi-3.5-vision-instruct model**](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)). This model was selected because we found that it was easy to use, relatively fast, able to read, and available as a NIM on [**build.nvidia.com**](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct).

This process will commandeer the Jupyter event loop and will render this notebook blocked, but feel free to test the endpoint out in a different notebook.


```python
# !vllm serve microsoft/Phi-3.5-vision-128k-instruct \

!vllm serve microsoft/phi-3.5-vision-instruct  \
    --trust-remote-code \
    --max_model_len 16384 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --port 9000
    # --chat-template ./phi3.jinja \

## NOTE: 
# - trust-remote-code enabled because this model incorporates some custom code modules not found in basic huggingface transformers
# - max-model-len set to ~16K since KV-cache takes up space and your GPU may start to run low if it's working with other processes
# - gpu memory capped to avoid potential conflicts with other local processes
# - more settings for more control, streamlines abstraction with other features offered by NIM-ified version
```

<hr>
<br>

<center><a href="https://www.nvidia.com/en-us/training/"><img src="https://dli-lms.s3.amazonaws.com/assets/general/DLI_Header_White.png" width="400" height="186" /></a></center>

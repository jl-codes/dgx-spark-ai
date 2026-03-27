# Running Your Own AI Supercomputer

**A hands-on curriculum for serving GPT-OSS 120B on the NVIDIA DGX Spark**

By the end of this guide, you'll have a 120-billion parameter AI model running on your desk, accessible via API, and powering your own AI coding assistant — all without touching the cloud.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.18.0-green)](https://github.com/vllm-project/vllm)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900)](https://developer.nvidia.com/cuda-toolkit)

**Navigation**: [Quick Start](#quick-start-tldr) | [Module 1](#module-1-understanding-your-hardware) | [Module 2](#module-2-setting-up-vllm) | [Module 3](#module-3-starting-your-first-model) | [Module 4](#module-4-your-first-conversation) | [Module 5](#module-5-building-with-python) | [🎉 Grand Finale](#-module-6-grand-finale--your-ai-coding-assistant)

---

## 🎯 What You'll Build

```
Hardware Check → vLLM Server → API Testing → Python Apps → 🎉 Cline Integration
   (15 min)        (30 min)      (30 min)     (30 min)         (45 min)
```

**The Goal**: In ~2.5 hours, you'll go from `nvidia-smi` to having your own AI coding assistant running on a 120B parameter model — locally, privately, with zero API costs.

**Prerequisites**:
- ✅ NVIDIA DGX Spark (or Grace Blackwell system)
- ✅ Ubuntu 24.04+ with CUDA 13.0
- ✅ Basic terminal comfort (cd, ls, curl, etc.)
- ✅ 60+ GB free disk space
- ✅ 2-3 hours of focused time

---

## Quick Start (TL;DR)

If you just want to get running and explore later:

```bash
# 1. Clone and enter
git clone https://github.com/jl-codes/dgx-spark-ai.git
cd dgx-spark-ai

# 2. Set up Python environment
make setup-venv

# 3. Start vLLM with GPT-OSS 120B (takes 60-90 seconds to load)
make serve

# 4. Test it (in another terminal)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello!"}]}'

# 5. Connect Cline CLI
make setup-cline

# 6. Use your AI coding assistant!
cline "Say hello and tell me what GPU I'm running on"
```

For the full learning experience with explanations and challenges, continue below! 👇

---

## Module 1: Understanding Your Hardware (15 min)

**🎓 Learning Objectives**: Understand what makes the DGX Spark special and why it changes the game for local AI

### 1.1 The DGX Spark: Your Personal AI Supercomputer

The NVIDIA DGX Spark isn't just a powerful computer — it's a fundamental rethink of how AI hardware works.

**Key Specs**:

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA GB10 (Grace-Blackwell architecture) |
| **CUDA Capability** | 12.1 (sm_121a) — latest generation |
| **Memory** | ~128 GB **unified** CPU+GPU |
| **CPU** | ARM64 (Grace, aarch64) |
| **CUDA Version** | 13.0 |

### 1.2 The Unified Memory Revolution

Here's what makes this different: **the CPU and GPU share the same physical memory pool**.

Traditional setup:
- GPU has 24GB VRAM
- CPU has 64GB RAM
- Moving data between them = bottleneck
- Can't load models larger than GPU VRAM

DGX Spark's unified memory:
- ✅ 128GB shared pool
- ✅ No CPU↔GPU transfer bottleneck
- ✅ Load models that would require multiple traditional GPUs
- ✅ Run GPT-OSS 120B (~65GB) with room to spare

This is why you can run a 120-billion parameter model on a desktop computer.

### 1.3 Check Your Hardware

Let's verify your system:

```bash
# Check GPU
nvidia-smi
```

You should see something like:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.95       Driver Version: 580.95       CUDA Version: 13.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GB10           On | 0000:0F:01:00.0  On |                  N/A |
+-------------------------------+----------------------+----------------------+
```

```bash
# Check total memory
free -h
```

You should see ~128GB total memory.

```bash
# Verify ARM64 architecture
uname -m
# Should output: aarch64
```

**🎯 Checkpoint**: You should see "GB10" (or similar Grace-Blackwell GPU) in `nvidia-smi` and ~128GB total memory.

<details>
<summary><b>💡 Deep Dive: Why Blackwell Changes Everything</b></summary>

The Grace-Blackwell architecture (GB10) introduces:

- **Unified memory architecture**: True CPU-GPU memory sharing via NVLink-C2C
- **CUDA Compute Capability 12.1**: New tensor core operations, FP4 support
- **ARM64 + NVIDIA GPU**: First consumer-grade ARM+CUDA system
- **mxfp4 microscaling**: Hardware-accelerated 4-bit floating point

This means:
1. Models quantized to mxfp4 run at near-FP16 quality but use 1/4 the memory
2. No data transfer latency between CPU and GPU memory spaces
3. Operating system can intelligently page memory
4. One 128GB pool > 24GB GPU + 64GB CPU separately
</details>

**⚡ Challenge 1**: Run `nvidia-smi dmon` and watch GPU utilization in real-time. What's your baseline GPU usage right now? (Press Ctrl+C to exit)

---

## Module 2: Setting Up vLLM (30 min)

**🎓 Learning Objectives**: Install vLLM, understand the project structure, and prepare to serve models

### 2.1 Clone and Explore the Project

```bash
# Clone the repository
git clone https://github.com/jl-codes/dgx-spark-ai.git
cd dgx-spark-ai

# Explore the structure
ls -la
```

**What you're looking at**:

```
dgx-spark-ai/
├── inference/          # Model serving scripts (vLLM)
│   ├── start_vllm_gptoss.sh    # Start GPT-OSS 120B
│   ├── stop_vllm.sh            # Stop vLLM
│   ├── health_check.sh         # Full pipeline health check
│   └── test_inference.py       # API smoke tests
├── training/           # Fine-tuning pipelines
│   ├── fine_tune.py            # QLoRA via Unsloth (fast)
│   ├── fine_tune_peft.py       # Standard PEFT/TRL
│   └── merge_lora.py           # Merge adapters
├── cline/             # Cline CLI integration
│   └── setup_cline_local.sh    # Auto-configure Cline
├── systemd/           # Production services
│   ├── vllm-server.service     # Auto-start on boot
│   └── vllm-watchdog.service   # Health monitoring
├── examples/          # Usage examples
│   ├── chat_completion.py      # Basic chat
│   ├── streaming_example.py    # Real-time streaming
│   └── batch_inference.py      # Concurrent requests
├── docs/              # Deep-dive documentation
├── Makefile           # Convenience commands
└── README.md          # You are here!
```

### 2.2 Create the Python Environment

vLLM requires Python 3.10+ with specific CUDA bindings. Let's set up a virtual environment:

```bash
make setup-venv
```

This installs:
- **vLLM 0.18.0+**: The inference engine
- **PyTorch 2.10+ with CUDA 13.0**: ML framework
- **OpenAI SDK**: For testing
- **All dependencies**: ~2GB total

```bash
# Activate the environment
source vllm-env/bin/activate

# Verify installation
vllm --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

You should see:
```
vLLM 0.18.0 (or newer)
PyTorch 2.x.x+cu130, CUDA: True
```

### 2.3 Understanding GPT-OSS 120B

**What is GPT-OSS?**
- OpenAI's first fully open-weight model
- 120 billion parameters
- Released with **mxfp4 quantization** (microscaling FP4)
- Ships at ~65GB instead of ~240GB (FP16)
- Competitive with much larger models due to efficient architecture

**What is mxfp4 quantization?**
- **4-bit floating point** with microscaling
- Groups weights into blocks, each with its own scale factor
- Near-FP16 quality at 1/4 the size
- Hardware-accelerated on Blackwell (GB10)
- This is why 120B parameters fit in 65GB

**Model location**: Models are cached at `~/.cache/huggingface/hub/` after first download.

### 2.4 Pre-download the Model (Optional)

The model auto-downloads on first run, but you can fetch it explicitly:

```bash
# This downloads ~65GB (takes 5-30 minutes depending on connection)
huggingface-cli download openai/gpt-oss-120b
```

**🎯 Checkpoint**: `vllm --version` returns 0.18.0+, and you have 60+ GB free disk space.

<details>
<summary><b>💡 Deep Dive: How vLLM Differs From Other Inference Engines</b></summary>

**vLLM vs. Ollama**:
- Ollama: Easy setup, single-user, GGUF models, llama.cpp backend
- vLLM: Production-grade, OpenAI API, multi-user, continuous batching

**vLLM vs. llama.cpp**:
- llama.cpp: CPU-friendly, GGUF format, great for smaller hardware
- vLLM: GPU-optimized, safetensors/HF format, better throughput

**vLLM vs. TGI (Text Generation Inference)**:
- TGI: HuggingFace's official server, good Docker support
- vLLM: Faster (PagedAttention), better batching, larger model support

**Why vLLM for the DGX Spark?**
1. **PagedAttention**: Manages GPU memory like an OS manages RAM (reduces waste by ~50%)
2. **Continuous batching**: Processes multiple requests simultaneously
3. **OpenAI compatibility**: Works with any OpenAI client
4. **Native mxfp4 support**: Optimized for Blackwell architecture
5. **Production-ready**: Used by major AI companies
</details>

**⚡ Challenge 2**: Check how much disk space the model uses (after download):
```bash
du -sh ~/.cache/huggingface/hub/models--openai--gpt-oss-120b
```
Compare with the theoretical size (120B params × 4 bits / 8 bits per byte).

---

## Module 3: Starting Your First Model (45 min)

**🎓 Learning Objectives**: Start the vLLM server, understand the startup process, monitor resources, and troubleshoot common issues

### 3.1 Starting vLLM

Time to bring GPT-OSS 120B to life! Open a terminal and run:

```bash
make serve
```

**What happens next** (60-90 seconds):
1. **Environment activation**: Loads the Python venv
2. **Library path setup**: Configures CUDA 13.0 libraries
3. **Tokenizer preparation**: Sets up Harmony tokenizer
4. **Model architecture detection**: Loads custom `GptOssForCausalLM` class
5. **Weight loading**: Reads 15 safetensor files (~65GB) into GPU memory
6. **Engine initialization**: Prepares vLLM scheduler and workers
7. **Server start**: OpenAI-compatible API on `localhost:8000`

You'll see a **lot** of log output. Here's what to look for:

```
INFO: Loading model weights...
Loading safetensors checkpoint shards:   7%|█▎              | 1/15 [00:02<00:31,  2.26s/it]
Loading safetensors checkpoint shards:  20%|███▋            | 3/15 [00:06<00:24,  2.08s/it]
...
INFO: Started server process [12345]
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**When you see "Started server process"** — you're ready! 🎉

### 3.2 Understanding the Startup Script

Let's peek under the hood. The `make serve` command runs `inference/start_vllm_gptoss.sh`, which does:

```bash
# Activate virtual environment
source vllm-env/bin/activate

# Set CUDA library paths (CUDA 13.0 for Blackwell)
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:..."

# Configure Harmony tokenizer cache
export TIKTOKEN_ENCODINGS_BASE="${HOME}/.cache/tiktoken-encodings/"

# Use system ptxas for Blackwell (sm_121a) support
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# Launch vLLM with optimized flags
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 32768 \
    --chat-template chat_template.jinja
```

### 3.3 Decoding the Flags

Each flag serves a critical purpose:

| Flag | What It Does | Why |
|------|--------------|-----|
| `--host 0.0.0.0` | Listen on all network interfaces | Allows access from other machines (optional) |
| `--port 8000` | API port | Standard HTTP port for vLLM |
| `--trust-remote-code` | Allow custom model code | GPT-OSS uses a custom architecture |
| `--enforce-eager` | Disable CUDA graphs | Saves ~10GB GPU memory (critical for 120B) |
| `--max-model-len 32768` | Context window limit | 32K tokens (longer = more KV cache memory) |
| `--chat-template` | Harmony chat format | GPT-OSS's analysis/commentary/final structure |

**Why `--enforce-eager` matters**:
- CUDA graphs = pre-compiled execution plans (faster but memory-hungry)
- With graphs: ~75GB memory needed
- Without graphs (eager mode): ~65GB memory needed
- Unified memory has 128GB, but OS + other processes use ~10-20GB
- **Eager mode lets us fit!**

### 3.4 Monitoring the Model Load

Open a **second terminal** and watch GPU memory fill up:

```bash
watch -n 2 nvidia-smi
```

You'll see:
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     12345      C   vllm serve                      67890MiB |
+-----------------------------------------------------------------------------+
```

Memory climbs from ~2GB → ~67GB over 60 seconds as weights load.

### 3.5 Common Startup Issues

**Problem 1: CUDA Out of Memory**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB
```

**Cause**: Another process is using GPU memory (often a zombie vLLM from before).

**Fix**:
```bash
# Stop all vLLM processes
make stop-force

# Wait for GPU to clear
sleep 3

# Restart
make serve
```

---

**Problem 2: Port Already in Use**
```
OSError: [Errno 98] Address already in use
```

**Cause**: Another service on port 8000 (maybe previous vLLM).

**Fix Option A** (kill existing):
```bash
make serve-force  # Kills existing and starts fresh
```

**Fix Option B** (use different port):
```bash
VLLM_PORT=8001 make serve
```

---

**Problem 3: Chat Template Not Found**
```
WARNING: chat_template.jinja not found, chat completions may not work
```

**Cause**: Model not fully downloaded.

**Fix**:
```bash
huggingface-cli download openai/gpt-oss-120b --force-download
make serve
```

---

**Problem 4: Slow Startup (>3 minutes)**

**First time**: Compiling CUDA kernels (normal, future runs are faster)

**Every time**: System may be swapping memory to disk.

**Check**:
```bash
free -h  # Look at "Swap" line
```

If swap usage is high, close other applications.

### 3.6 Verifying Success

In another terminal:

```bash
# Quick check
make check
# Output: "vLLM server is healthy on port 8000"

# Or manually test
curl http://localhost:8000/v1/models
```

You should get:
```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-oss-120b",
      "object": "model",
      "owned_by": "vllm",
      "max_model_len": 32768
    }
  ]
}
```

**🎯 Checkpoint**: Server shows "Started server process" and `curl http://localhost:8000/v1/models` returns JSON with the model ID.

<details>
<summary><b>💡 Deep Dive: vLLM Architecture</b></summary>

**Inside vLLM**:

```
┌─────────────────────────────────────────────────┐
│  FastAPI Server (port 8000)                     │
│  └─ /v1/models, /v1/chat/completions, etc.      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  LLM Engine                                      │
│  ├─ Request queue                                │
│  ├─ Continuous batching scheduler                │
│  └─ KV cache manager (PagedAttention)           │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  Workers (GPU processes)                         │
│  ├─ Model loaded in GPU memory                   │
│  ├─ Tokenizer (Harmony for GPT-OSS)             │
│  └─ CUDA kernels (attention, sampling, etc.)    │
└──────────────────────────────────────────────────┘
```

**PagedAttention**:
- KV cache is divided into "pages" (like OS virtual memory)
- Pages can be swapped, shared between requests, etc.
- Reduces memory waste from padding by ~50%
- Critical for fitting large context windows

**Continuous Batching**:
- Traditional: Wait for all requests in batch to finish
- vLLM: Add/remove requests dynamically as they complete
- Result: Better GPU utilization, lower latency

**Why startup takes 60-90s**:
1. Load 65GB weights into memory: ~30s
2. Initialize CUDA contexts: ~10s
3. Compile Triton kernels (first time): ~20-40s
4. Build KV cache structures: ~5s
5. Start HTTP server: ~1s
</details>

**⚡ Challenge 3**: Time your startup! Run `make serve` and measure how long from command execution to "Started server process". Compare with the 60-90s benchmark. Faster or slower? (Hint: `time make serve` won't work because the server keeps running — use a stopwatch!)

---

## Module 4: Your First Conversation (30 min)

**🎓 Learning Objectives**: Interact with the model via API, understand request/response formats, and explore GPT-OSS's unique reasoning capabilities

### 4.1 The OpenAI-Compatible API

vLLM exposes the same API as OpenAI's cloud service. This means:

✅ Any OpenAI client works out-of-the-box  
✅ Familiar interface, local compute  
✅ Easy migration from cloud to local  
✅ Swap `base_url` and you're done  

**Endpoints available**:
- `GET /v1/models` — List available models
- `POST /v1/chat/completions` — Chat with the model
- `POST /v1/completions` — Raw text completion
- `POST /v1/embeddings` — Generate embeddings (if model supports)

### 4.2 List Available Models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

**Response**:
```json
{
    "object": "list",
    "data": [
        {
            "id": "openai/gpt-oss-120b",
            "object": "model",
            "owned_by": "vllm",
            "created": 1711507200,
            "max_model_len": 32768,
            "permission": [...]
        }
    ]
}
```

**What this tells you**:
- **Model ID**: `openai/gpt-oss-120b` (use this in requests)
- **Max context**: 32,768 tokens
- **Owner**: `vllm` (local server, not OpenAI)

### 4.3 Your First Chat Message

Let's ask the model a question:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "Explain unified memory in one sentence."}
    ],
    "max_tokens": 100
  }' | python3 -m json.tool
```

**Response** (abbreviated):
```json
{
  "id": "chat-f7a8b9c0...",
  "object": "chat.completion",
  "created": 1711507812,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Unified memory allows the CPU and GPU to access the same physical memory pool without copying data between separate memory spaces.",
        "reasoning": "[Analysis] The user wants a concise explanation of unified memory architecture. [Commentary] I should focus on the key benefit: elimination of data transfer overhead. [Final] One sentence capturing the essence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 31,
    "total_tokens": 49
  }
}
```

**Breaking it down**:

| Field | Meaning |
|-------|---------|
| `choices[0].message.content` | The actual answer |
| `choices[0].message.reasoning` | **GPT-OSS specialty**: Internal chain-of-thought |
| `usage.prompt_tokens` | Your input (18 tokens) |
| `usage.completion_tokens` | Model's output (31 tokens) |
| `finish_reason` | Why it stopped (`stop` = natural end, `length` = hit max_tokens) |

### 4.4 Understanding GPT-OSS's Reasoning Format

GPT-OSS is a **reasoning model** that generates structured thought channels:

**Three channels**:
1. **Analysis**: Understanding the request
2. **Commentary**: Meta-reasoning about the approach
3. **Final**: The actual response to the user

This is similar to OpenAI's o1/o3 models, but fully open and local!

**Why this matters**:
- You can see *why* the model chose its response
- Helps debug unexpected outputs
- Enables advanced prompting techniques
- Great for educational use

**The `reasoning` field** contains this internal monologue (when present).

### 4.5 Streaming Responses

For a better user experience, stream tokens as they're generated:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
    "stream": true
  }'
```

**Output** (Server-Sent Events):
```
data: {"id":"chat-123","object":"chat.completion.chunk","created":1711507900,"model":"openai/gpt-oss-120b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chat-123","object":"chat.completion.chunk","created":1711507900,"model":"openai/gpt-oss-120b","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: {"id":"chat-123","object":"chat.completion.chunk","created":1711507900,"model":"openai/gpt-oss-120b","choices":[{"index":0,"delta":{"content":","},"finish_reason":null}]}

data: {"id":"chat-123","object":"chat.completion.chunk","created":1711507900,"model":"openai/gpt-oss-120b","choices":[{"index":0,"delta":{"content":" 2"},"finish_reason":null}]}

...

data: [DONE]
```

Watch tokens appear in real-time! This is the **Server-Sent Events (SSE)** protocol.

### 4.6 Experimenting with Parameters

Try different generation settings:

```bash
# More creative (higher temperature)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Write a haiku about AI."}],
    "temperature": 0.9,
    "max_tokens": 50
  }' | python3 -m json.tool

# Deterministic (temperature 0)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0,
    "max_tokens": 10
  }' | python3 -m json.tool
```

**Key parameters**:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0.0 - 2.0 | Higher = more random/creative |
| `top_p` | 0.0 - 1.0 | Nucleus sampling (0.9 = top 90% probability mass) |
| `max_tokens` | 1 - 32768 | Maximum response length |
| `presence_penalty` | -2.0 - 2.0 | Discourage repetition |
| `frequency_penalty` | -2.0 - 2.0 | Penalize token frequency |

**🎯 Checkpoint**: You can send messages via curl and receive responses. Streaming works. You understand the response structure.

<details>
<summary><b>💡 Deep Dive: Token Generation and Performance Metrics</b></summary>

**How token generation works**:

1. **Tokenization**: "Hello world" → `[15496, 1917]`
2. **Embedding**: Token IDs → vectors
3. **Transformer layers**: 120B parameters process vectors
4. **Next token prediction**: Probability distribution over vocab
5. **Sampling**: Pick next token based on temperature/top_p
6. **Repeat**: Until `</s>` or max_tokens

**Performance metrics**:

- **TTFT (Time To First Token)**: Latency until first output token
  - For GPT-OSS 120B on GB10: ~500-1500ms
  - Depends on prompt length (longer = more prefill work)

- **Tokens/second**: Generation speed after first token
  - For GPT-OSS 120B on GB10: ~5-15 tokens/sec
  - Varies with context length, batch size

- **Throughput vs. Latency**: Single user = optimize latency, Multiple users = optimize throughput

**Why GPT-OSS 120B is "slow" compared to cloud**:
- Cloud APIs use tensor parallelism (multiple GPUs)
- GPT-4/Claude use speculative decoding, optimized infrastructure
- 120B params on 1 GPU is impressive but compute-bound

**When local is worth it**:
- ✅ Zero per-token cost (free after hardware)
- ✅ Complete privacy (data never leaves your machine)
- ✅ No rate limits
- ✅ Customizable (fine-tuning, prompt templates)
- ✅ Learning and experimentation

</details>

**⚡ Challenge 4**: Ask the model about itself! Run:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "What GPU am I running on? Be specific about the architecture."}],
    "max_tokens": 150
  }' | python3 -m json.tool
```

Does it know about the GB10/Grace-Blackwell? Check the `reasoning` field to see its thought process!

---

## Module 5: Building with Python (30 min)

**🎓 Learning Objectives**: Use the OpenAI Python SDK, explore example patterns, and build your first application

### 5.1 Setting Up the OpenAI SDK

The OpenAI Python SDK works seamlessly with vLLM. Just point it at your local endpoint:

```python
from openai import OpenAI

# Create client pointing to local vLLM
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-vllm"  # Can be anything (vLLM doesn't check)
)

# List models
models = client.models.list()
print(f"Available: {models.data[0].id}")

# Chat completion
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### 5.2 Example 1: Basic Chat Completion

Let's run the first example:

```bash
python3 examples/chat_completion.py
```

**What it does**:
- Connects to local vLLM
- Sends a chat message
- Displays the response
- Shows token usage

**Code walkthrough** (`examples/chat_completion.py`):

```python
#!/usr/bin/env python3
"""Basic chat completion example using the OpenAI SDK."""

from openai import OpenAI

def main():
    # Connect to local vLLM server
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="local-vllm"
    )
    
    # Send a message
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain what you are in one sentence."}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    # Extract and display
    message = response.choices[0].message
    print(f"Response: {message.content}")
    
    # Show token usage
    usage = response.usage
    print(f"\nTokens used: {usage.total_tokens} "
          f"(prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens})")

if __name__ == "__main__":
    main()
```

**Key concepts**:
- **System message**: Sets the AI's behavior/personality
- **User message**: Your actual request
- **Assistant message**: The AI's response (in multi-turn conversations)

### 5.3 Example 2: Streaming Responses

For better UX, stream tokens as they arrive:

```bash
python3 examples/streaming_example.py
```

**Code walkthrough** (`examples/streaming_example.py`):

```python
#!/usr/bin/env python3
"""Streaming chat completion example."""

from openai import OpenAI
import sys

def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="local-vllm"
    )
    
    # Enable streaming with stream=True
    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": "Write a short story about an AI (3 sentences)."}
        ],
        max_tokens=200,
        temperature=0.8,
        stream=True  # <-- Key difference
    )
    
    print("Response: ", end="", flush=True)
    
    # Process chunks as they arrive
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
    
    print()  # Newline at end

if __name__ == "__main__":
    main()
```

**Why streaming matters**:
- User sees output immediately (not waiting 10+ seconds)
- Better perceived performance
- Can cancel early if output goes off-track
- Essential for chat UIs

### 5.4 Example 3: Batch Processing

Process multiple requests concurrently:

```bash
python3 examples/batch_inference.py
```

**Code walkthrough** (`examples/batch_inference.py`):

```python
#!/usr/bin/env python3
"""Batch inference with concurrent requests."""

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time

def query_model(client, prompt):
    """Send a single query."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response.choices[0].message.content

def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="local-vllm"
    )
    
    # Batch of prompts
    prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "What is the speed of light?",
        "Who wrote Hamlet?",
        "What is the largest planet?",
    ]
    
    # Process concurrently
    start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(
            lambda p: query_model(client, p),
            prompts
        ))
    elapsed = time.time() - start
    
    # Display results
    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result}\n")
    
    print(f"Processed {len(prompts)} requests in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(prompts):.2f}s per request")

if __name__ == "__main__":
    main()
```

**Why batch processing matters**:
- vLLM's **continuous batching** processes multiple requests simultaneously
- 5 concurrent requests ≠ 5× the time (more like 2-3× due to batching)
- Better GPU utilization
- Critical for multi-user scenarios

**🎯 Checkpoint**: All three examples run successfully. You understand basic chat, streaming, and batching patterns.

<details>
<summary><b>💡 Deep Dive: vLLM Request Lifecycle</b></summary>

**What happens when you call `client.chat.completions.create()`:**

1. **Client side**:
   - OpenAI SDK formats request as JSON
   - Sends HTTP POST to `http://localhost:8000/v1/chat/completions`

2. **vLLM server receives**:
   - FastAPI endpoint validates request
   - Applies chat template (Harmony format for GPT-OSS)
   - Tokenizes input using Harmony tokenizer

3. **Request enters queue**:
   - vLLM's scheduler adds request to queue
   - Waits for available GPU slots

4. **Prefill phase** (first token):
   - Entire prompt processed in parallel
   - Generates KV cache for prompt
   - Samples first output token
   - **This is the TTFT latency**

5. **Decode phase** (subsequent tokens):
   - One token generated per step
   - Auto-regressive: each token depends on previous
   - KV cache grows with each token
   - **This is the tokens/sec throughput**

6. **Batching magic**:
   - Multiple requests in prefill/decode simultaneously
   - Scheduler fills GPU compute capacity
   - Requests at different stages batched together
   - **This is why concurrent requests are efficient**

7. **Response sent**:
   - If streaming: SSE chunks as each token generated
   - If not streaming: Full response after `</s>` or max_tokens

**KV Cache Management**:
- Each request's KV cache stored in PagedAttention blocks
- Blocks can be shared (same prompt prefix = shared blocks)
- Blocks freed when request finishes
- **This is why vLLM uses less memory than naive implementations**

</details>

**⚡ Challenge 5**: Modify `examples/chat_completion.py` to:
1. Ask the model to write a haiku about AI hardware
2. Save the response to a file called `haiku.txt`
3. Print the token usage

Hint: Use `open('haiku.txt', 'w')` and `f.write(content)`.

---

## 🎉 Module 6: Grand Finale — Your AI Coding Assistant (45 min)

**🎓 Learning Objectives**: Connect Cline CLI to your local model and experience AI-powered development

### 6.1 What is Cline?

[Cline](https://github.com/cline/cline) is an AI coding assistant that can:

✅ Read and write files in your project  
✅ Execute terminal commands  
✅ Browse the web for documentation  
✅ Reason through complex multi-step tasks  
✅ Use 44+ AI providers (including "OpenAI Compatible")  

**Why Cline + Local vLLM is powerful**:
- 🔒 **Complete privacy**: Your code never leaves your machine
- 💰 **Zero cost**: No per-token charges (after hardware)
- ⚡ **No rate limits**: Generate as much as you need
- 🎓 **Learning**: See exactly how the AI thinks (reasoning traces)
- 🔧 **Customizable**: Fine-tune the model for your domain

### 6.2 Installing Cline CLI

If you don't have Cline installed:

```bash
# Install via npm
npm install -g @cline/cli

# Verify installation
cline --version
```

### 6.3 Auto-Configuration

The easiest way to connect Cline to your local vLLM:

```bash
make setup-cline
```

**What this does**:

1. **Checks** if vLLM is running on port 8000
2. **Fetches** the model ID from `/v1/models`
3. **Configures** Cline's OpenAI provider settings:
   - Provider: `openai`
   - API Key: `local-vllm` (arbitrary, vLLM doesn't check)
   - Model ID: `openai/gpt-oss-120b`
   - Base URL: `http://localhost:8000/v1`
4. **Verifies** the connection works

**Manual configuration** (if you prefer):

```bash
cline auth \
    --provider openai \
    --apikey "local-vllm" \
    --modelid "openai/gpt-oss-120b" \
    --baseurl "http://localhost:8000/v1"
```

**Verify configuration**:

```bash
cline config
```

Look for:
```
actModeApiProvider: openai
actModeOpenAiModelId: openai/gpt-oss-120b
openAiBaseUrl: http://localhost:8000/v1
planModeApiProvider: openai
planModeOpenAiModelId: openai/gpt-oss-120b
```

### 6.4 Your First Cline Task

Let's start simple:

```bash
cline "What GPU am I running on?"
```

Watch Cline:
1. Send request to your local GPT-OSS 120B
2. Parse the response
3. Display it with formatting

**Example output**:
```
╔══════════════════════════════════════════════════════╗
║  Cline CLI                                          ║
╚══════════════════════════════════════════════════════╝

Analyzing environment...

You're running on an NVIDIA GB10 (Grace-Blackwell) GPU with unified 
memory architecture, CUDA capability 12.1, and approximately 128GB of 
shared CPU+GPU memory. This is part of the NVIDIA DGX Spark platform.

(Response generated in 3.2s using openai/gpt-oss-120b)
```

🎉 **Congratulations!** You just used your own 120B parameter AI model as a coding assistant!

### 6.5 Real Coding Tasks

Now let's get practical:

#### Task 1: Generate a Python Function

```bash
cline "Write a Python function to calculate the Fibonacci sequence up to n terms. Include docstring and type hints."
```

Cline will:
- Generate the code
- Explain what it does
- Optionally save to a file (if you ask)

#### Task 2: Work with Files

```bash
cline "Read the Makefile and explain what the 'serve' target does"
```

Cline will:
- Use `read_file` tool to open Makefile
- Analyze the content
- Explain the shell commands

#### Task 3: Multi-Step Task

```bash
cline "Create a new Python file called 'hello.py' that prints 'Hello from GPT-OSS 120B!' when run, then show me how to execute it."
```

Cline will:
1. Create the file
2. Write the Python code
3. Provide the execution command
4. Optionally run it for you

### 6.6 Interactive Mode (The Full Experience)

Launch the Cline TUI:

```bash
cline
```

This opens a full terminal user interface with:

```
┌─────────────────────────────────────────────────────┐
│  Cline CLI v2.11.0                                  │
│  Connected to: openai/gpt-oss-120b (local)          │
├─────────────────────────────────────────────────────┤
│  [Chat Interface]                                   │
│                                                     │
│  You: Read examples/chat_completion.py and add      │
│       error handling                                 │
│                                                     │
│  Cline: I'll analyze the file and add try/except    │
│         blocks...                                    │
│         [... thinking ...]                           │
│         [Uses read_file tool]                        │
│         [Proposes changes]                           │
│         Would you like me to apply these changes?    │
│                                                     │
│  You: yes                                            │
│                                                     │
│  Cline: [Applies changes with write_to_file]        │
│         Done! The file now has error handling.       │
├─────────────────────────────────────────────────────┤
│  [File Tree]              [Command Output]          │
│  ├─ examples/             $ python3 test.py         │
│  │  ├─ chat_completion.py ✓ Tests passed           │
│  │  ├─ streaming_...      ...                       │
│  └─ inference/                                       │
└─────────────────────────────────────────────────────┘
```

**Features**:
- **Chat interface**: Conversational interaction
- **File tree**: See project structure
- **Command execution**: Run shell commands
- **Tool use**: Read files, write files, search, etc.

### 6.7 Plan vs. Act Modes

Cline has two operating modes:

**Act Mode** (default):
- Executes tasks immediately
- Best for simple, clear requests
- Example: "Create a file called test.txt"

**Plan Mode**:
- Thinks through complex tasks first
- Creates a plan, gets your approval
- Then executes step by step
- Example: "Refactor this codebase to use async/await"

**Try Plan Mode**:

```bash
cline  # Enter interactive mode
# Type: /plan
# Then describe a complex task
```

Example plan mode session:
```
You: /plan
     Add comprehensive error handling to all examples/ scripts

Cline: I'll analyze each file and propose changes. Here's my plan:

       1. Read all Python files in examples/
       2. Identify places needing try/except
       3. Add error handling for:
          - Network errors (OpenAI API calls)
          - File I/O errors
          - JSON parsing errors
       4. Add logging for better debugging
       5. Test each modified file
       
       Does this plan look good? (yes/no)

You: yes

Cline: Executing step 1...
       [Proceeds through the plan]
```

### 6.8 Advanced: Using Reasoning Traces

Because GPT-OSS 120B includes reasoning traces, you can see *why* Cline makes decisions:

```bash
# In Cline's config, enable showing reasoning
cline config set showReasoning true

# Now when you ask Cline something complex:
cline "What's the best way to optimize the vLLM startup script?"
```

You'll see:
```
[Reasoning]
[Analysis] The user is asking about optimization opportunities in 
the startup script. I should consider memory usage, startup time, 
and error handling.

[Commentary] Key areas to examine: library path setup (can we cache?), 
model loading flags (eager vs. graph tradeoffs), health check logic.

[Final] I'll read the script and suggest specific improvements...

[Response]
Let me analyze the startup script for optimization opportunities...
```

**This is incredibly valuable for learning** how the AI approaches problems!

**🎯 Checkpoint**: Cline successfully generates code using your local GPT-OSS 120B model. You can run both one-shot commands and interactive mode.

<details>
<summary><b>💡 Deep Dive: How Cline Uses the OpenAI API</b></summary>

**Cline's architecture**:

```
┌───────────────────────────────────────────────┐
│  Cline CLI (Node.js)                          │
│  ├─ User interface (TUI or one-shot)          │
│  ├─ Conversation manager                      │
│  └─ Tool system (read_file, execute, etc.)    │
└────────────────┬──────────────────────────────┘
                 │
                 ▼ HTTP POST
┌────────────────────────────────────────────────┐
│  vLLM Server (:8000)                           │
│  └─ /v1/chat/completions                       │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│  GPT-OSS 120B Model                            │
│  └─ Returns: reasoning + content                │
└────────────────────────────────────────────────┘
```

**A typical Cline request**:

```json
{
  "model": "openai/gpt-oss-120b",
  "messages": [
    {
      "role": "system",
      "content": "You are Cline, an AI coding assistant. You can read files, write code, and execute commands..."
    },
    {
      "role": "user",
      "content": "Read examples/chat_completion.py and add error handling"
    }
  ],
  "temperature": 0.7,
  "stream": true
}
```

**Cline's system prompt** tells GPT-OSS:
- What tools are available
- How to format tool calls
- When to ask for user confirmation
- How to structure responses

**Why local + Cline is special**:
- **Privacy**: Code never sent to cloud
- **Customization**: Can modify system prompt, add tools
- **Cost**: Generate 10,000 tokens? Free. (vs. $3+ on GPT-4)
- **Learning**: Full visibility into API calls and reasoning
- **No censorship**: Model isn't filtered for corporate safety

**Cost comparison** (10,000 tokens/day for a month):

| Provider | Cost/Month |
|----------|------------|
| GPT-4 Turbo | ~$100 |
| Claude 3 Opus | ~$150 |
| Local GPT-OSS 120B | $0 (electricity: ~$5) |

</details>

**⚡ Challenge 6**: Ask Cline to create a Python script that:
1. Queries the vLLM API at `http://localhost:8000/v1/models`
2. Extracts the model name and max context length
3. Prints them formatted nicely

Then run the generated script!

```bash
cline "Create a Python script called model_info.py that fetches and displays info from the local vLLM API at http://localhost:8000/v1/models. Use the requests library."

# After Cline creates it:
python3 model_info.py
```

**⚡ Challenge 7** (Advanced): Use Cline in Plan Mode to add streaming support to the script from Challenge 6:

```bash
cline
# Type: /plan
# Task: Modify model_info.py to also test streaming chat completions
```

Watch Cline:
1. Create a plan
2. Get your approval
3. Modify the file
4. Test it

---

## Module 7: Going Further

You've completed the core curriculum! 🎉 Here's where to go next:

### 7.1 Fine-Tuning Your Own Models

Want to specialize GPT-OSS for your specific use case?

**Quick start**:
```bash
# Prepare your dataset (JSONL format)
# Each line: {"instruction": "...", "response": "..."}

# Fine-tune with QLoRA (uses Unsloth for 2x speed)
make train DATASET=your_data.jsonl

# Or use standard PEFT/TRL
make train-peft DATASET=your_data.jsonl
```

**Deep dive**: [Training Guide](docs/TRAINING_GUIDE.md)

Topics covered:
- Dataset preparation
- QLoRA vs. full fine-tuning
- Hyperparameter tuning
- Merging LoRA adapters
- Exporting to GGUF for Ollama
- Evaluation and validation

### 7.2 Production Deployment

Running vLLM as a systemd service for auto-start and monitoring:

```bash
# Install services
make install-services

# Services installed:
# - vllm-server.service: Auto-starts on boot
# - vllm-watchdog.timer: Health check every 2 minutes
# - vllm-watchdog.service: Auto-restart if unhealthy

# Check status
make status

# View logs
journalctl --user -u vllm-server -f
```

**Deep dive**: [Systemd Services](systemd/README.md)

### 7.3 Switching Models

Try different models for different tasks:

```bash
# Fast 7B model for quick iterations
bash inference/switch_model.sh Qwen/Qwen2.5-7B-Instruct

# 32B coding specialist
bash inference/switch_model.sh Qwen/Qwen2.5-Coder-32B-Instruct

# Back to GPT-OSS 120B
bash inference/switch_model.sh openai/gpt-oss-120b
```

**Model comparison**:

| Model | Parameters | Memory | Context | Best For |
|-------|-----------|--------|---------|----------|
| **GPT-OSS 120B** | 120B (mxfp4) | ~65GB | 32K | Best quality, reasoning |
| Qwen2.5-7B-Instruct | 7B | ~15GB | 32K | Fast testing, general use |
| Qwen2.5-Coder-32B | 32B | ~24GB | 32K | Code generation |
| Llama-3.1-8B | 8B | ~17GB | 8K | General purpose |
| Llama-3.1-70B | 70B (quantized) | ~45GB | 8K | High quality (fits on GB10!) |

### 7.4 Troubleshooting

Running into issues? Check the troubleshooting guide:

**[Full Troubleshooting Guide](docs/TROUBLESHOOTING.md)**

**Quick diagnostics**:
```bash
# Run comprehensive health check
make health

# Check GPU
nvidia-smi

# Check vLLM processes
pgrep -fa "vllm serve"

# Check port
ss -tlnp | grep 8000

# View logs
journalctl --user -u vllm-server --no-pager -n 50
```

**Most common issues**:

1. **CUDA OOM**: Another process using GPU → `make stop-force`
2. **Port in use**: Another server on 8000 → `make serve-force`
3. **Slow inference**: Expected for 120B (5-15 tokens/sec)
4. **Cline not connecting**: Verify base URL includes `/v1`

### 7.5 Complete Documentation

**Deep-dive resources**:

- 📖 [**Complete Tutorial**](docs/LESSON_VLLM_DGX_SPARK.md) — 10,000-word guide with architecture deep-dives
- 🎯 [**Training Guide**](docs/TRAINING_GUIDE.md) — End-to-end fine-tuning walkthrough
- 🏗️ [**Architecture**](docs/ARCHITECTURE.md) — System design and component breakdown
- 🔧 [**Troubleshooting**](docs/TROUBLESHOOTING.md) — Solutions to common issues

### 7.6 Advanced Topics

**For the curious**:

- **Custom chat templates**: Modify GPT-OSS's reasoning format
- **Multi-GPU setup**: Tensor parallelism (if you have multiple DGX Sparks!)
- **Quantization comparison**: mxfp4 vs. int8 vs. int4 vs. FP16
- **Inference optimization**: Batching strategies, speculative decoding
- **Fine-tuning for Cline**: Train a model specifically for coding tasks
- **Tool use**: Adding custom tools to GPT-OSS via function calling

---

## 📋 Quick Reference

**Most Common Commands**:

```bash
# Start vLLM
make serve              # Normal start
make serve-force        # Kill existing + restart

# Health checks
make health             # Full pipeline test
make check              # Quick API check

# Stop vLLM
make stop               # Graceful shutdown
make stop-force         # Force kill

# Cline
make setup-cline        # Configure Cline

# Training
make train DATASET=data.jsonl           # QLoRA fine-tuning
make merge-lora ADAPTER=... OUTPUT=...  # Merge LoRA weights

# Services
make install-services   # Auto-start on boot
make status             # Check services
```

**Full Command Reference**:

```
$ make help

╔══════════════════════════════════════════════════════════════╗
║  dgx-spark-ai — vLLM + Training on NVIDIA DGX Spark        ║
╚══════════════════════════════════════════════════════════════╝

  serve                  Start vLLM server with GPT-OSS 120B
  serve-force            Force-restart vLLM server (kills existing)
  stop                   Stop all vLLM processes
  stop-force             Force-kill all vLLM processes
  health                 Run full pipeline health check
  check                  Quick check if vLLM is responding
  test-api               Run API smoke tests
  train                  Fine-tune with QLoRA via Unsloth (DATASET=path)
  train-peft             Fine-tune with standard PEFT/TRL (DATASET=path)
  merge-lora             Merge LoRA adapter into base model
  export-gguf            Export model to GGUF format
  setup-cline            Configure Cline CLI to use local vLLM
  install-services       Install systemd user services
  uninstall-services     Remove systemd user services
  status                 Show status of all services
  setup-venv             Create Python venv and install vLLM
  setup-training-deps    Install training dependencies
```

---

## 🏆 What You've Accomplished

**Congratulations!** 🎉 You've completed the curriculum. You now know how to:

- ✅ Understand the DGX Spark's unified memory architecture
- ✅ Install and configure vLLM for production use
- ✅ Serve a 120-billion parameter model locally
- ✅ Interact with the model via OpenAI-compatible API
- ✅ Build Python applications using the OpenAI SDK
- ✅ Use Cline as your AI coding assistant with your own model
- ✅ Deploy to production with systemd services
- ✅ Troubleshoot common issues
- ✅ Run everything on your hardware with zero cloud dependency

### The Bottom Line

You now have:
- 🔒 **Complete privacy**: Your data never leaves your machine
- 💰 **Zero recurring costs**: No per-token charges
- ⚡ **No rate limits**: Generate as much as you need
- 🎓 **Deep understanding**: You know how every piece works
- 🚀 **Production-ready setup**: Auto-start, monitoring, health checks

### Next Steps

- **Fine-tune** a model on your own data for domain specialization
- **Experiment** with different models (Qwen, Llama, etc.)
- **Build** applications using the local API
- **Share** your learnings with the community
- **Contribute** improvements back to this project

---

## 🌟 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    NVIDIA DGX Spark                          │
│                   (128GB Unified Memory)                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  User Layer                                          │    │
│  │  ├─ Cline CLI (AI coding assistant)                 │    │
│  │  ├─ Python Apps (OpenAI SDK)                        │    │
│  │  └─ Direct API calls (curl, any HTTP client)        │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                       │
│                      │ HTTP (OpenAI-compatible API)          │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  vLLM Server (Port 8000)                            │    │
│  │  ├─ FastAPI: /v1/models, /v1/chat/completions      │    │
│  │  ├─ Continuous batching scheduler                   │    │
│  │  ├─ PagedAttention KV cache manager                 │    │
│  │  └─ Systemd service (auto-start, watchdog)          │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                       │
│                      │ Python / CUDA                         │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Model Layer                                         │    │
│  │  ├─ GPT-OSS 120B (mxfp4, ~65GB)                     │    │
│  │  ├─ Harmony tokenizer (chat template)               │    │
│  │  └─ Reasoning: analysis/commentary/final            │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                       │
│                      │ CUDA / Tensor ops                     │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  NVIDIA GB10 (Grace-Blackwell)                      │    │
│  │  ├─ CUDA 13.0, Compute Capability 12.1              │    │
│  │  ├─ mxfp4 hardware acceleration                     │    │
│  │  ├─ Unified memory architecture                     │    │
│  │  └─ ~128GB CPU+GPU shared memory pool               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Training Pipeline (Optional)                        │    │
│  │  ├─ QLoRA fine-tuning (Unsloth / PEFT)             │    │
│  │  ├─ Dataset: your_data.jsonl                        │    │
│  │  ├─ Merge LoRA adapters                             │    │
│  │  └─ Export GGUF → Ollama                            │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

Found a bug? Have an improvement? Contributions welcome!

```bash
# Fork the repo
gh repo fork jl-codes/dgx-spark-ai

# Create a branch
git checkout -b my-improvement

# Make changes, commit, push
git commit -am "Add feature X"
git push origin my-improvement

# Open a PR
gh pr create
```

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgments

- **NVIDIA**: For the incredible DGX Spark hardware
- **vLLM team**: For the best inference engine
- **OpenAI**: For GPT-OSS 120B and open weights
- **Cline team**: For the amazing coding assistant
- **Open source community**: For making all of this possible

---

**Built with ❤️ by the community. Run AI your way.**

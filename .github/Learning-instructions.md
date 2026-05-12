# Learning Instructions

For each topic listed below, create a dedicated folder containing:

1. **Learning Material** (`README.md`) — A beginner-friendly guide covering:
   - Core concepts with clear, concise explanations
   - Simple, runnable code examples for each concept
   - Diagrams or visuals where helpful
   - A glossary of key terms

2. **Project Tutorial** (`PROJECT.md`) — A hands-on, step-by-step project that reinforces the concepts, including:
   - Prerequisites and environment setup
   - Incremental build steps (each step should compile/run successfully)
   - Expected output or screenshots at each milestone
   - Common pitfalls and troubleshooting tips

**Guidelines:**
- Write for absolute beginners — assume no prior knowledge of the topic
- Keep examples minimal and focused on one concept at a time
- Include short examples and diagrams where possible 
- Use consistent naming conventions and folder structure across all topics

***Presentation***
Create a simple webpage for viewing the projects 

## Topics

### Project 1: Build an LLM Playground

LLM Overview and Foundations
    - Pre-Training
    - Data collection (manual crawling, Common Crawl)
    - Data cleaning (RefinedWeb, Dolma, FineWeb)
    - Tokenization (e.g., BPE)
    - Architecture (neural networks, Transformers, GPT family, DeepSeek, Qwen, Gemma)
    - Text generation (greedy and beam search, top-k, top-p)
Post-Training
    - SFT
    - RL and RLHF (verifiable tasks, reward models, PPO, etc.)
Evaluation
    - Traditional metrics
    - Task-specific benchmarks
    - Human evaluation and leaderboards
Chatbots' Overall Design

### Project 2: Build a Customer Support Chatbot using RAGs and Prompt Engineering

Overview of Adaptation Techniques
Finetuning
    - Parameter-efficient fine-tuning (PEFT)
    - Adapters and LoRA
Prompt Engineering
    - Few-shot and zero-shot prompting
    - Chain-of-thought prompting
    - Role-specific and user-context prompting
RAGs Overview
Retrieval
    - Document parsing (rule-based, AI-based) and chunking strategies
    - Indexing (keyword, full-text, knowledge-based, vector-based, embedding models)
Generation
    - Search methods (exact and approximate nearest neighbor)
    - Prompt engineering for RAGs
RAFT: Training technique for RAGs
Evaluation (context relevance, faithfulness, answer correctness)
RAGs' Overall Design

### Project 3: Build an "Ask-the-Web" Agent similar to Perplexity with Tool Calling

Agents Overview
    - Agents vs. agentic systems vs. LLMs
    - Agency levels (e.g., workflows, multi-step agents)
Workflows
    - Prompt chaining
    - Routing
    - Parallelization (sectioning, voting)
    - Reflection
    - Orchestration-worker
Tools
    - Tool calling
    - Tool formatting
    - Tool execution
    - MCP
Multi-Step Agents
    - Planning autonomy
    - ReACT
    - Reflexion, ReWOO, etc.
    - Tree search for agents
Multi-Agent Systems
    - Challenges
    - Use-cases
    - A2A protocol
Agent Evaluation

### Project 4: Build "Deep Research" Capability with Web Search and Reasoning Models

Reasoning and Thinking LLMs
    - Overview of reasoning models like OpenAI's "o" family and DeepSeek-R1
Inference-time Techniques
    - Inference-time scaling
    - CoT prompting
    - Parallel sampling
    - Sequential sampling
    - Tree of Thoughts (ToT)
    - Search against a verifier
Training-time Techniques
    - SFT on reasoning data (e.g., STaR)
    - Reinforcement learning with a verifier
    - Reward modeling (ORM, PRM)
    - Self-refinement
    - Internalizing search (e.g., Meta-CoT)
Local Deployment

### Project 5: Build a Multi-modal Generation Agent

Overview of Image and Video Generation
    - VAE
    - GANs
    - Auto-regressive models
    - Diffusion models
Text-to-Image (T2I)
    - Data preparation
    - Diffusion architectures (U-Net, DiT)
    - Diffusion training (forward process, backward process)
    - Diffusion sampling
    - Evaluation (image quality, diversity, image-text alignment, IS, FID, and CLIP score)
Text-to-Video (T2V)
    - Latent-diffusion modeling (LDM) and compression networks
    - Data preparation (filtering, standardization, video latent caching)
    - DiT architecture for videos
    - Large-scale training challenges
    - T2V's overall system

### Capstone Project: Ship a Portfolio-Ready AI Project from Idea to Demo

    - Choose: pick your own idea, or start from a curated list
    - Build: implement using techniques from the course
    - Iterate: get real-time feedback from the instructor as you build
    - Optional Demo: present your project on final demo day

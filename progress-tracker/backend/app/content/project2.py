PROJECT_2 = {
    "number": 2,
    "title": "Build a Customer Support Chatbot using RAGs",
    "description": "Build a Customer Support Chatbot using RAGs",
    "topics": [
        {
            "title": "Finetuning",
            "content": """## Finetuning

Finetuning is the process of taking a pre-trained language model and further training it on a domain-specific dataset so that it learns specialized knowledge, tone, or behavior. In the context of a customer support chatbot, finetuning can teach the model your company's product details, support policies, and communication style.

### When to Finetune vs. RAG vs. Prompt Engineering

Choosing the right approach depends on your use case, budget, and data:

| Approach | Best For | Cost | Data Needed |
|---|---|---|---|
| **Prompt Engineering** | Quick prototyping, simple tasks | Low | None (just good prompts) |
| **RAG** | Knowledge-heavy tasks, frequently changing data | Medium | A document corpus |
| **Finetuning** | Style/tone adaptation, consistent behavior, niche domains | High | Hundreds to thousands of examples |

### Decision Matrix

Ask yourself these questions:

1. **Does the model already know the information?** If yes, prompt engineering may suffice.
2. **Does the information change frequently?** If yes, RAG is better since you update documents, not the model.
3. **Do you need a specific tone or behavior pattern?** Finetuning excels here.
4. **What is your budget?** Finetuning requires GPU compute; RAG requires a vector database; prompt engineering is nearly free.

### Cost/Benefit Analysis

- **Finetuning** gives you the most control but costs the most in compute and data preparation. A finetuned model can hallucinate if it hasn't seen relevant data during training.
- **RAG** grounds responses in actual documents, reducing hallucination, and lets you update knowledge without retraining.
- **Prompt engineering** is the fastest to iterate on but can be fragile with complex tasks.

In practice, many production chatbots **combine all three**: a finetuned base model, enhanced with RAG for knowledge retrieval, guided by carefully engineered prompts. This layered approach maximizes accuracy, consistency, and maintainability.
""",
            "children": [
                {
                    "title": "PEFT",
                    "content": """## Parameter-Efficient Fine-Tuning (PEFT)

PEFT refers to a family of techniques that fine-tune only a small subset of a model's parameters while keeping the majority frozen. This dramatically reduces the computational cost, memory requirements, and storage overhead of fine-tuning large language models.

### Why Full Finetuning Is Impractical

A model like LLaMA-2 70B has **70 billion parameters**. Full finetuning requires:

- Multiple high-end GPUs (e.g., 4-8x A100 80GB)
- Storing a full copy of gradients and optimizer states
- Significant training time and cost (thousands of dollars per run)

For most teams and use cases, this is simply not feasible.

### Types of PEFT Methods

| Method | Description | Trainable Params |
|---|---|---|
| **LoRA** | Injects low-rank matrices into attention layers | ~0.1-1% of total |
| **Prefix Tuning** | Prepends learnable tokens to each layer's input | Very small |
| **Prompt Tuning** | Learns soft prompt embeddings prepended to input | Minimal |
| **Adapters** | Inserts small trainable bottleneck layers between transformer blocks | ~1-5% |
| **IA3** | Learns scaling vectors for keys, values, and FFN activations | Extremely small |

### How PEFT Works Conceptually

Instead of updating all weights `W`, PEFT methods learn a small **delta** `dW` such that the effective weight becomes `W + dW`. The key insight is that `dW` can be represented in a much lower-dimensional space, making training efficient without sacrificing much performance.

### Benefits of PEFT

- **Lower cost**: Train on a single consumer GPU in many cases
- **Faster iteration**: Experiments run in hours instead of days
- **Multiple adapters**: Store tiny adapter weights for different tasks on top of the same base model
- **Reduced catastrophic forgetting**: Freezing most parameters preserves the model's general capabilities

PEFT methods have become the standard approach for customizing LLMs in production, enabling teams of all sizes to build specialized models without massive infrastructure investments.
""",
                },
                {
                    "title": "Adapters and LoRA",
                    "content": """## Adapters and LoRA

### LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is the most popular PEFT method. The core idea is simple: instead of updating a full weight matrix `W` (of size `d x d`), LoRA learns two small matrices `A` (size `d x r`) and `B` (size `r x d`), where `r` (the rank) is much smaller than `d`.

The effective weight becomes:

```
W' = W + A * B
```

Since `r` is typically 4, 8, 16, or 32, the number of trainable parameters drops by **orders of magnitude**.

### Rank Selection Guidelines

| Rank (`r`) | Use Case | Trainable Params |
|---|---|---|
| 4-8 | Simple style/tone adaptation | Very few |
| 16-32 | Domain-specific knowledge | Moderate |
| 64-128 | Complex task adaptation | More (still far less than full) |

Higher rank = more capacity but more compute. Start small and increase only if performance is lacking.

### QLoRA: Quantized LoRA

QLoRA combines LoRA with 4-bit quantization of the base model. This lets you finetune a 70B parameter model on a **single 48GB GPU** by loading the base weights in 4-bit precision while training LoRA adapters in full precision.

### Python Example with PEFT + Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,  # QLoRA
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                     # rank
    lora_alpha=32,            # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

LoRA adapters are saved as small files (typically 10-100 MB) that can be loaded on top of any copy of the base model, making deployment and versioning straightforward.
""",
                },
            ],
        },
        {
            "title": "Prompt Engineering",
            "content": """## Prompt Engineering

Prompt engineering is the practice of designing and refining the text inputs (prompts) given to a language model to elicit the most accurate, relevant, and useful responses. It is the fastest and most cost-effective way to control LLM behavior without any model training.

### Why Prompt Engineering Matters

Large language models are general-purpose: they can write poetry, solve math, or debug code. The **prompt** is what steers the model toward your specific task. A well-crafted prompt can be the difference between a helpful customer support response and a generic, unhelpful one.

### Prompt Anatomy

Modern chat-based LLMs use a structured message format with three roles:

| Role | Purpose | Example |
|---|---|---|
| **System** | Sets behavior, personality, constraints, and context | "You are a helpful customer support agent for Acme Corp..." |
| **User** | The customer's actual message or question | "How do I reset my password?" |
| **Assistant** | The model's response (or a pre-filled example) | "To reset your password, visit..." |

```
+-------------------------------------------+
|  System Prompt                            |
|  - Role definition                        |
|  - Behavioral constraints                 |
|  - Output format instructions             |
+-------------------------------------------+
|  User Message                             |
|  - The actual query or input              |
+-------------------------------------------+
|  Assistant Response                       |
|  - Generated by the model                 |
+-------------------------------------------+
```

### Key Principles

1. **Be specific**: Vague prompts yield vague answers. State exactly what you want.
2. **Provide context**: Give the model the information it needs to answer correctly.
3. **Constrain the output**: Specify format, length, and style expectations.
4. **Iterate**: Prompt engineering is experimental -- test, evaluate, and refine.

### Prompt Engineering in a Chatbot Context

For a customer support chatbot, prompts should define the agent's persona, the scope of topics it can handle, escalation procedures, and the tone of communication. The system prompt becomes the "personality file" of your chatbot, and getting it right is essential for a professional, trustworthy user experience.
""",
            "children": [
                {
                    "title": "Few-shot/zero-shot",
                    "content": """## Few-Shot and Zero-Shot Prompting

### Zero-Shot Prompting

Zero-shot prompting means asking the model to perform a task **without any examples**. You rely entirely on the model's pre-trained knowledge and clear instructions.

```
System: You are a customer support classifier. Classify the following
        message into one of: billing, technical, general.

User: I can't log into my account after changing my password.

Assistant: technical
```

### Few-Shot Prompting (In-Context Learning)

Few-shot prompting provides **examples** of the desired input-output behavior directly in the prompt. The model learns the pattern from the examples and applies it to new inputs.

```
System: Classify customer messages. Here are examples:

User: I was charged twice for my subscription.
Assistant: billing

User: The app crashes when I open settings.
Assistant: technical

User: What are your business hours?
Assistant: general

User: My invoice shows the wrong amount.
Assistant: billing
```

### Example Selection Best Practices

- **Diversity**: Cover all categories/edge cases in your examples
- **Relevance**: Choose examples similar to expected real inputs
- **Consistency**: Use a uniform format across all examples
- **Order matters**: Place the most representative examples first
- **Balance**: Include roughly equal numbers of examples per category

### Formatting Guidelines

| Practice | Why It Helps |
|---|---|
| Use clear delimiters between examples | Prevents the model from blending examples together |
| Keep examples concise | Saves token budget for the actual task |
| Match the format of the desired output | The model mimics the pattern it sees |
| Include edge cases | Teaches the model how to handle ambiguity |

### When to Use Which

- **Zero-shot**: When the task is straightforward and the model performs well without examples
- **Few-shot (1-3 examples)**: When you need consistent formatting or category definitions
- **Many-shot (5+ examples)**: When the task is nuanced and requires pattern demonstration

Few-shot prompting is one of the most powerful techniques because it requires no training -- just well-chosen examples in the prompt.
""",
                },
                {
                    "title": "Chain-of-thought",
                    "content": """## Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting encourages the model to **show its reasoning step by step** before arriving at a final answer. This dramatically improves performance on tasks requiring logic, multi-step reasoning, or complex decision-making.

### The Core Technique

The simplest CoT trigger is adding **"Let's think step by step"** to your prompt:

```
User: A customer bought a $50 plan on March 15 and wants a prorated
      refund for cancelling on March 25. Our billing cycle is monthly.
      How much is the refund?

Assistant: Let's think step by step:
1. The billing cycle is 30 days (monthly).
2. The customer used the plan for 10 days (March 15-25).
3. Cost per day = $50 / 30 = $1.67
4. Days used = 10, so amount used = $16.70
5. Refund = $50 - $16.70 = $33.30

The refund amount is **$33.30**.
```

### Self-Consistency

Self-consistency extends CoT by generating **multiple reasoning paths** and selecting the most common answer. This is especially useful for ambiguous customer queries:

```
System: Think through this customer issue using three different
        reasoning approaches, then provide your final answer based
        on the majority conclusion.
```

### Structured Reasoning Template

For customer support scenarios, structured CoT works well:

```
System: When handling customer issues, follow this reasoning process:
  1. IDENTIFY: What is the customer's core issue?
  2. CONTEXT: What relevant policies or information apply?
  3. OPTIONS: What are the possible resolutions?
  4. RECOMMEND: What is the best resolution and why?
  5. RESPOND: Draft a clear, empathetic response.
```

### When to Use Chain-of-Thought

- **Complex calculations**: Refunds, prorations, pricing
- **Policy interpretation**: Applying nuanced rules to specific situations
- **Troubleshooting**: Walking through diagnostic steps
- **Escalation decisions**: Determining if a ticket needs human review

CoT prompting adds tokens to the output, increasing latency and cost, so use it selectively for tasks where reasoning quality matters more than speed.
""",
                },
                {
                    "title": "Role-specific prompting",
                    "content": """## Role-Specific Prompting

Role-specific prompting uses the **system prompt** to define a detailed persona for the language model. This sets the tone, boundaries, knowledge scope, and behavioral guidelines that shape every response the chatbot generates.

### Persona Definition

A well-defined persona includes:

- **Identity**: Who is the agent? (name, company, department)
- **Tone**: Professional, friendly, formal, casual
- **Scope**: What topics can it help with? What should it refuse?
- **Escalation rules**: When should it hand off to a human?

### System Prompt Template

```
You are Alex, a customer support agent for TechCorp.

## Your Role
- Help customers with billing, technical issues, and account management
- Be professional, empathetic, and concise
- Always greet the customer and ask clarifying questions if needed

## Guidelines
- Never share internal system details or pricing negotiations
- If unsure about an answer, say: "Let me connect you with a specialist"
- For billing disputes over $500, escalate to a human agent
- Always confirm the customer's issue is resolved before closing

## Tone
- Warm but professional
- Use the customer's name when available
- Avoid jargon; explain technical concepts simply

## Restrictions
- Do NOT provide legal advice
- Do NOT share other customers' information
- Do NOT make promises about future product features
```

### Guardrails

Guardrails are explicit constraints in the system prompt that prevent the model from going off-script:

| Guardrail Type | Example |
|---|---|
| **Topic restriction** | "Only discuss topics related to our products and services" |
| **Safety boundary** | "If a user expresses distress, provide the support hotline number" |
| **Data protection** | "Never ask for or confirm full credit card numbers" |
| **Hallucination prevention** | "If you don't know the answer, say so -- do not guess" |

### Tone Control

You can fine-tune tone by providing examples of desired vs. undesired responses:

```
GOOD: "I understand how frustrating that must be. Let me look into this for you."
BAD:  "That's not our problem. Check the FAQ."
```

Role-specific prompting is the foundation of a well-behaved chatbot. Invest time in crafting your system prompt -- it is the single most impactful piece of your chatbot's design.
""",
                },
            ],
        },
        {
            "title": "Retrieval",
            "content": """## Retrieval in RAG

Retrieval-Augmented Generation (RAG) enhances a language model by fetching relevant documents from an external knowledge base before generating a response. The retrieval component is the backbone of any RAG system.

### RAG Pipeline Overview

```
                    INGESTION PIPELINE
  +----------+    +--------+    +----------+    +-----------+
  | Documents |--->| Parse  |--->|  Chunk   |--->|  Embed &  |
  | (PDF,HTML)|    | & Clean|    | Strategy |    |  Index    |
  +----------+    +--------+    +----------+    +-----------+
                                                      |
                                                      v
                    QUERY PIPELINE              +----------+
  +--------+    +--------+    +---------+      | Vector   |
  | User   |--->| Embed  |--->| Search  |<-----| Database |
  | Query  |    | Query  |    | & Rank  |      +----------+
  +--------+    +--------+    +---------+
                                   |
                                   v
                            +-----------+    +----------+
                            | Augmented  |--->| Generate |
                            | Prompt     |    | Response |
                            +-----------+    +----------+
```

### Why Retrieval Reduces Hallucination

LLMs generate text based on patterns learned during training. When asked about specific facts, they may produce plausible-sounding but incorrect information (hallucinations). Retrieval solves this by:

1. **Grounding**: Providing actual source documents the model can reference
2. **Freshness**: Including up-to-date information not present in the training data
3. **Specificity**: Supplying domain-specific details the model may not have memorized
4. **Verifiability**: Enabling citation of sources for transparency

### Benefits for Customer Support

- **Always current**: Product docs, pricing, and policies stay up to date
- **Accurate answers**: Responses are based on your actual knowledge base
- **Auditable**: You can trace every answer back to its source document
- **Scalable**: Adding new knowledge means adding new documents, not retraining

The retrieval component transforms a general-purpose LLM into a knowledgeable, domain-specific assistant that can answer questions with the authority and accuracy of your documentation.
""",
            "children": [
                {
                    "title": "Document parsing",
                    "content": """## Document Parsing

Document parsing is the first step in the RAG ingestion pipeline. It converts raw documents (PDFs, HTML pages, markdown files) into clean, structured text that can be chunked and embedded.

### Parsing Libraries

| Library | Format | Strengths |
|---|---|---|
| **PyPDF2 / PyMuPDF** | PDF | Fast, extracts text and metadata from PDFs |
| **unstructured** | PDF, HTML, DOCX, PPTX | Multi-format, handles complex layouts |
| **BeautifulSoup** | HTML | Flexible HTML parsing and cleaning |
| **markdown-it-py** | Markdown | Parses markdown into structured elements |
| **docx2txt** | DOCX | Simple Word document text extraction |

### Chunking Strategies

After parsing, documents must be split into chunks for embedding. The chunking strategy significantly affects retrieval quality.

| Strategy | Description | Best For |
|---|---|---|
| **Fixed-size** | Split every N characters/tokens with overlap | Simple, predictable |
| **Recursive** | Split by paragraph, then sentence, then character | General-purpose (LangChain default) |
| **Semantic** | Split at topic/meaning boundaries using embeddings | High-quality retrieval |
| **Document-aware** | Split by headers, sections, or logical structure | Structured docs (manuals, FAQs) |

### Chunking Best Practices

- **Chunk size**: 256-1024 tokens is typical. Smaller chunks = more precise retrieval; larger chunks = more context.
- **Overlap**: Use 10-20% overlap between chunks to avoid splitting important context across boundaries.
- **Metadata**: Attach source file name, page number, section title, and timestamp to each chunk.

### Metadata Extraction

Good metadata improves retrieval filtering and citation:

```python
chunk = {
    "text": "To reset your password, go to Settings > Security...",
    "metadata": {
        "source": "user_guide_v3.pdf",
        "page": 42,
        "section": "Account Management",
        "last_updated": "2025-01-15",
    }
}
```

High-quality parsing and chunking is foundational -- poor input at this stage cascades into poor retrieval and poor answers downstream.
""",
                },
                {
                    "title": "Indexing",
                    "content": """## Indexing

Indexing is the process of converting text chunks into vector embeddings and storing them in a vector database for efficient similarity search. This is the bridge between parsed documents and fast retrieval at query time.

### Vector Databases

| Database | Type | Strengths |
|---|---|---|
| **FAISS** | Library (in-memory) | Fast, free, great for prototyping |
| **ChromaDB** | Embedded DB | Easy API, good for small-medium scale |
| **Pinecone** | Managed cloud | Scalable, production-ready, managed infrastructure |
| **Weaviate** | Self-hosted / cloud | Hybrid search, GraphQL API |
| **Qdrant** | Self-hosted / cloud | Rich filtering, fast performance |

### Embedding Models

Embeddings convert text into dense numerical vectors that capture semantic meaning:

| Model | Dimensions | Provider |
|---|---|---|
| **text-embedding-3-small** | 1536 | OpenAI |
| **text-embedding-3-large** | 3072 | OpenAI |
| **all-MiniLM-L6-v2** | 384 | Sentence-Transformers (free) |
| **BGE-large-en** | 1024 | BAAI (free) |

### Similarity Search

At query time, the user's question is embedded using the **same model**, and the vector database finds the most similar document chunks using distance metrics like cosine similarity.

### Python Example: Building an Index with ChromaDB

```python
import chromadb
from openai import OpenAI

# Initialize clients
chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_client = OpenAI()

# Create a collection
collection = chroma_client.create_collection(name="support_docs")

# Prepare documents
documents = [
    "To reset your password, navigate to Settings > Security > Reset Password.",
    "Refund requests must be submitted within 30 days of purchase.",
    "Our API rate limit is 100 requests per minute per API key.",
]

# Generate embeddings
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=documents,
)
embeddings = [item.embedding for item in response.data]

# Add to vector database
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=documents,
    embeddings=embeddings,
)

# Query
results = collection.query(
    query_texts=["How do I change my password?"],
    n_results=2,
)
print(results['documents'])
```

A well-built index with the right embedding model and database is critical for fast, accurate retrieval.
""",
                },
            ],
        },
        {
            "title": "Generation",
            "content": """## Generation in RAG

The generation step is where the language model produces a response using the retrieved documents as context. This is the final stage of the RAG pipeline, and its quality depends on how well the retrieved context is injected into the prompt.

### Context Injection

Retrieved documents are inserted into the prompt alongside the user's question, typically in a structured format:

```
System: You are a customer support assistant. Answer the user's question
using ONLY the provided context. If the context doesn't contain the
answer, say "I don't have that information."

Context:
[1] To reset your password, go to Settings > Security > Reset Password.
    You will receive a confirmation email within 5 minutes.
[2] Password requirements: minimum 8 characters, one uppercase letter,
    one number, one special character.

User: How do I reset my password?
```

### Faithfulness vs. Creativity Tradeoff

This is one of the most important design decisions in RAG:

| Setting | Faithfulness | Creativity | Use Case |
|---|---|---|---|
| **High faithfulness** | Strict adherence to retrieved docs | Low | Factual Q&A, policy questions |
| **Balanced** | Uses docs as primary source, fills gaps naturally | Medium | General support |
| **High creativity** | Uses docs as inspiration, generates freely | High | Brainstorming, suggestions |

For customer support, **high faithfulness** is almost always preferred. You want the chatbot to answer based on your documentation, not make up plausible-sounding answers.

### Controlling Faithfulness

- **Prompt instructions**: Explicitly tell the model to only use provided context
- **Temperature**: Use low temperature (0.0-0.3) for factual responses
- **Instruction phrasing**: "Answer ONLY based on the context above" vs. "Use the context to help answer"
- **No-answer handling**: Instruct the model to say "I don't know" rather than guess

### Generation Best Practices

- Number your context chunks so the model can cite sources
- Place context before the question for better attention
- Limit context to the most relevant chunks (quality over quantity)
- Include metadata (source, date) so the model can provide attribution
""",
            "children": [
                {
                    "title": "Search methods",
                    "content": """## Search Methods

Search methods determine how your RAG system finds relevant documents for a given query. Different methods have different strengths, and production systems often combine multiple approaches.

### Dense Retrieval

Dense retrieval uses **neural embeddings** to represent both queries and documents as dense vectors. Similarity is computed via cosine similarity or dot product. This is the standard approach in most RAG systems.

- **Strengths**: Captures semantic meaning, handles synonyms and paraphrases
- **Weaknesses**: Can miss exact keyword matches, requires embedding model

### Sparse Retrieval (BM25)

BM25 is a classic information retrieval algorithm based on **term frequency** and **inverse document frequency**. It excels at exact keyword matching.

- **Strengths**: Fast, interpretable, excellent for keyword queries
- **Weaknesses**: Misses semantic similarity ("car" vs. "automobile")

### Hybrid Search

Hybrid search **combines dense and sparse retrieval** to get the best of both worlds:

```
final_score = alpha * dense_score + (1 - alpha) * sparse_score
```

Where `alpha` is a tunable weight (typically 0.5-0.7 favoring dense). This ensures both semantic matches and exact keyword matches are captured.

### Re-Ranking

Re-ranking is a second-stage process that takes the top-K results from initial retrieval and **re-scores them** using a more powerful model:

```
Initial retrieval (fast) --> Top 50 results --> Re-ranker (accurate) --> Top 5 results
```

Popular re-ranking services include **Cohere Rerank**, **Jina Reranker**, and cross-encoder models from Sentence-Transformers.

### MMR: Maximal Marginal Relevance

MMR balances **relevance** and **diversity** in results. Without MMR, you might retrieve five documents that all say the same thing. MMR ensures the returned documents cover different aspects of the query:

```
MMR = argmax[lambda * Sim(doc, query) - (1 - lambda) * max(Sim(doc, selected_docs))]
```

| Method | Speed | Semantic Understanding | Exact Match |
|---|---|---|---|
| Dense | Medium | Excellent | Weak |
| BM25 | Fast | Weak | Excellent |
| Hybrid | Medium | Good | Good |
| Hybrid + Re-rank | Slower | Excellent | Good |

For customer support, **hybrid search with re-ranking** provides the best overall quality.
""",
                },
                {
                    "title": "Prompt engineering for RAGs",
                    "content": """## Prompt Engineering for RAGs

Prompt engineering for RAG systems focuses on effectively integrating retrieved context into the prompt and handling edge cases gracefully. It builds on general prompt engineering but adds RAG-specific considerations.

### Context Window Management

LLMs have a limited context window (e.g., 4K, 8K, 128K tokens). Managing it wisely is critical:

- **Prioritize**: Place the most relevant chunks first (models attend to early context more)
- **Truncate**: Set a maximum number of chunks (typically 3-5 for support queries)
- **Summarize**: For long documents, summarize chunks before injecting them
- **Token budget**: Reserve tokens for the system prompt, user query, and generated response

```
Token Budget Example (8K context window):
- System prompt:      ~500 tokens
- Retrieved context:  ~4000 tokens (3-5 chunks)
- User query:         ~200 tokens
- Response budget:    ~3300 tokens
```

### Citation Formatting

Instruct the model to cite its sources so users can verify answers:

```
System: When answering, cite the source documents using [1], [2], etc.
        Format your answer as:
        - Direct answer to the question
        - Sources: list the document numbers used

Context:
[1] Password reset: Go to Settings > Security... (user_guide.pdf, p.42)
[2] Account lockout: After 5 failed attempts... (security_policy.pdf, p.7)
```

### Handling No Results

When retrieval returns no relevant documents, the model must handle it gracefully:

```
System: If the provided context does not contain information relevant
to the user's question:
1. Acknowledge that you don't have specific information on this topic
2. Suggest the user contact support@company.com for further help
3. Do NOT make up an answer
```

### Anti-Hallucination Prompting

Use explicit instructions to prevent the model from going beyond the retrieved context:

- "Answer ONLY based on the context provided above"
- "If the answer is not in the context, respond with: I don't have that information"
- "Do not use your general knowledge for factual claims"

These prompt patterns are essential for building a trustworthy RAG chatbot that users can rely on for accurate information.
""",
                },
            ],
        },
        {
            "title": "RAFT",
            "content": """## RAFT: Retrieval Augmented Fine-Tuning

RAFT (Retrieval Augmented Fine-Tuning) is a training methodology that combines the benefits of RAG and finetuning. Instead of choosing between them, RAFT **trains the model to be better at using retrieved documents** by including both relevant and irrelevant (distractor) documents during training.

### Core Concept

Standard finetuning teaches a model to answer questions from its parameters. Standard RAG provides context at inference time. RAFT bridges the gap by training the model **with retrieval in mind**:

1. For each training example, provide the question plus a set of documents
2. Include the **correct** document (oracle) mixed with **distractor** documents
3. Train the model to identify and use the correct document while ignoring distractors
4. Some examples deliberately exclude the oracle, teaching the model to say "I don't know"

### Training Data Format

```
Input:
  Question: What is the refund policy for annual plans?
  Document 1 (distractor): Our API supports REST and GraphQL...
  Document 2 (oracle): Annual plans can be refunded within 60 days...
  Document 3 (distractor): System requirements include 4GB RAM...

Output:
  Annual plans can be refunded within 60 days of purchase.
  The refund is prorated based on the remaining months. [Source: Document 2]
```

### When to Use RAFT vs. Standard RAG

| Scenario | Recommendation |
|---|---|
| Frequently changing knowledge base | Standard RAG (no retraining needed) |
| Fixed domain with noisy retrieval | RAFT (better at filtering distractors) |
| Need for high faithfulness to sources | RAFT (trained to cite correctly) |
| Limited compute budget | Standard RAG (no training required) |
| Model struggles with long context | RAFT (learns to extract relevant info) |

### Benefits of RAFT

- **Robustness**: The model learns to handle noisy retrieval results
- **Citation accuracy**: Training with source attribution improves citation quality
- **Reduced hallucination**: Exposure to distractors teaches the model to be cautious
- **Better extraction**: The model learns to find and synthesize relevant information from long contexts

RAFT is especially valuable when your retrieval system isn't perfect (which is always the case) and you need the model to be resilient to irrelevant search results.
""",
            "children": [],
        },
        {
            "title": "Evaluation",
            "content": """## Evaluation of RAG Systems

Evaluating a RAG system requires measuring both the quality of retrieval and the quality of generation. The **RAGAS** (Retrieval Augmented Generation Assessment) framework provides a comprehensive set of metrics for this purpose.

### RAGAS Framework

RAGAS evaluates RAG pipelines across four key dimensions:

| Metric | What It Measures | Scale |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | 0-1 |
| **Answer Relevance** | Does the answer address the question asked? | 0-1 |
| **Context Precision** | Are the retrieved documents relevant to the question? | 0-1 |
| **Context Recall** | Did retrieval find all the necessary information? | 0-1 |

### Faithfulness

Faithfulness measures whether the generated answer is **grounded in the retrieved context**. A faithful answer doesn't contain claims that aren't supported by the provided documents.

- Score of 1.0: Every claim in the answer can be traced to the context
- Score of 0.0: The answer is entirely fabricated or unsupported

### Answer Relevance

Answer relevance checks if the response actually **answers the question**. A response might be faithful to the context but irrelevant to what was asked.

### Context Precision and Recall

These metrics evaluate the **retrieval component**:

- **Context Precision**: What fraction of the retrieved documents are actually relevant?
  - High precision = few irrelevant documents retrieved
- **Context Recall**: What fraction of the relevant documents were retrieved?
  - High recall = most relevant documents were found

### Evaluation Workflow

```
1. Create a test dataset with:
   - Questions
   - Ground truth answers
   - Ground truth relevant documents

2. Run your RAG pipeline on each question

3. Compute RAGAS metrics:
   - Compare retrieved docs vs. ground truth docs (precision/recall)
   - Compare generated answer vs. ground truth (relevance)
   - Check generated answer vs. retrieved context (faithfulness)

4. Iterate on your pipeline based on which metrics are lowest
```

### Practical Tips

- Start with **50-100 test questions** covering common customer queries
- Include edge cases: questions with no answer, ambiguous questions, multi-part questions
- Re-evaluate after every pipeline change (embedding model, chunking strategy, prompt updates)
- Track metrics over time to detect regression
""",
            "children": [],
        },
        {
            "title": "RAGs' Overall Design",
            "content": """## RAGs' Overall Design

A production RAG system consists of two main pipelines -- the **ingestion pipeline** and the **query pipeline** -- plus supporting infrastructure for monitoring, caching, and maintenance.

### Full Architecture

```
                        INGESTION PIPELINE
  +---------+   +-------+   +-------+   +--------+   +---------+
  | Source   |-->| Parse |-->| Chunk |-->| Embed  |-->| Vector  |
  | Docs     |   |       |   |       |   |        |   | Database|
  +---------+   +-------+   +-------+   +--------+   +---------+
      |                                                    |
      v                                                    v
  +---------+                                        +---------+
  | Metadata|                                        | Index   |
  | Store   |                                        | (FAISS/ |
  +---------+                                        | Chroma) |
                                                     +---------+
                        QUERY PIPELINE                    |
  +---------+   +-------+   +--------+   +---------+    |
  | User    |-->| Embed |-->| Search |<--| Re-rank |<---+
  | Query   |   | Query |   |        |   |         |
  +---------+   +-------+   +--------+   +---------+
                                |                  
                                v                  
                          +---------+   +----------+
                          | Build   |-->| Generate |
                          | Prompt  |   | Response |
                          +---------+   +----------+
```

### Production Considerations

#### Caching

- **Query cache**: Store responses for frequently asked questions to reduce latency and cost
- **Embedding cache**: Cache embeddings for repeated queries to avoid redundant API calls
- **Semantic cache**: Use similarity matching to serve cached answers for semantically similar queries

#### Monitoring

Track these metrics in production:

| Metric | Why It Matters |
|---|---|
| **Latency** (P50, P95, P99) | User experience degrades above 3-5 seconds |
| **Retrieval hit rate** | Percentage of queries with relevant results |
| **User satisfaction** | Thumbs up/down, CSAT scores |
| **Fallback rate** | How often the bot says "I don't know" or escalates |
| **Token usage** | Cost monitoring for embedding and generation API calls |

#### Document Freshness

- Schedule regular re-ingestion of updated documents
- Implement webhooks or watchers for real-time document updates
- Version your document index to enable rollbacks

#### Guardrails and Safety

- Input filtering: Block prompt injection attempts and abusive content
- Output filtering: Check responses for PII leakage, policy violations
- Rate limiting: Prevent abuse and control costs

A well-architected RAG system is more than just retrieval and generation -- it includes the operational infrastructure needed to run reliably, affordably, and safely in production.
""",
            "children": [],
        },
    ],
}

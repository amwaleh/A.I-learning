PROJECT_3 = {
    "number": 3,
    "title": "Build an Ask-the-Web Agent",
    "description": "Build an Ask-the-Web Agent",
    "topics": [
        {
            "title": "Workflows",
            "content": """## Agentic Workflows

Agentic workflows describe the **design patterns** that govern how LLMs interact with tools, data, and each other to accomplish complex tasks. Rather than a single prompt-response exchange, workflows orchestrate multiple steps — each potentially involving reasoning, tool use, or delegation.

Anthropic's taxonomy (from their *"Building Effective Agents"* blog post) identifies five foundational patterns:

| Pattern | When to Use |
|---|---|
| **Prompt Chaining** | Sequential tasks where each step's output feeds the next |
| **Routing** | Input must be classified and dispatched to a specialist |
| **Parallelization** | Independent subtasks that can run concurrently |
| **Reflection** | Output needs self-critique and iterative improvement |
| **Orchestration-Worker** | Complex tasks requiring decomposition and delegation |

### Choosing the Right Pattern

The key insight is to **use the simplest pattern that solves your problem**. Start with prompt chaining for straightforward pipelines. Add routing when you need specialization. Use parallelization for throughput. Layer in reflection when quality matters more than speed. Resort to orchestration-worker only for genuinely complex, multi-faceted tasks.

```mermaid
graph LR
    A["Single LLM Call<br/>(simplest)"] --> B["Prompt Chain"] --> C["Router"] --> D["Parallel"] --> E["Reflection"] --> F["Orchestrator<br/>(most complex)"]
```

### Workflows vs. Autonomous Agents

A **workflow** is a predefined, deterministic arrangement of LLM calls — you control the flow. An **autonomous agent**, by contrast, decides its own next step at each turn. In practice most production systems are workflows with agentic *components*, not fully autonomous agents. This hybrid approach gives you reliability (from the workflow skeleton) with flexibility (from agentic decision-making at key nodes).

Understanding these patterns is essential before building your Ask-the-Web agent, because your agent will combine several of them: routing user queries, chaining search results through summarization, and reflecting on answer quality.

<!-- checkpoint: I understand the five agentic workflow patterns and when to use each -->
""",
            "children": [
                {
                    "title": "Prompt chaining",
                    "content": """## Prompt Chaining

Prompt chaining is the simplest agentic pattern: you break a task into sequential steps, where each LLM call receives the output of the previous one. Between steps you can insert **gates** — programmatic checks that validate intermediate results before proceeding.

### Why Chain?

A single monolithic prompt often fails on complex tasks because it overloads the model's context and reasoning capacity. Chaining lets each step focus on one concern, improving accuracy and making the pipeline debuggable.

### Architecture

```mermaid
graph LR
    Input --> Step1["Step 1: Extract"]
    Step1 --> G1{"Gate ✓"}
    G1 -->|Pass| Step2["Step 2: Transform"]
    G1 -->|Fail| Retry["Retry / Abort"]
    Step2 --> G2{"Gate ✓"}
    G2 -->|Pass| Step3["Step 3: Format"]
    G2 -->|Fail| Retry
    Step3 --> Output
```

Gates can be:
- **Schema validators** (JSON Schema, Pydantic)
- **LLM-as-judge** (ask another LLM if the output is valid)
- **Programmatic checks** (length, required fields, regex)

### Python Example

```python
from openai import OpenAI

client = OpenAI()

def chain(query: str) -> str:
    # Step 1: Extract key entities
    step1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Extract the key entities from: {query}"}]
    )
    entities = step1.choices[0].message.content

    # Gate: must contain at least one entity
    if len(entities.strip()) < 3:
        raise ValueError("No entities extracted")

    # Step 2: Generate search queries from entities
    step2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Generate 3 search queries for: {entities}"}]
    )
    queries = step2.choices[0].message.content

    # Step 3: Synthesize final answer
    step3 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Summarize findings for '{query}' "
                              f"using these queries: {queries}"}]
    )
    return step3.choices[0].message.content
```

### Best Practices

- Keep each step **focused on a single transformation**
- Always add validation gates between steps to catch errors early
- Log intermediate outputs for debugging
- Consider cost — each step is a separate API call
- Use cheaper models for simple transformation steps, reserve powerful models for reasoning-heavy steps
""",
                },
                {
                    "title": "Routing",
                    "content": """## Routing

Routing is the pattern of **classifying an input and dispatching it to a specialized handler**. Instead of one general-purpose prompt trying to do everything, a router directs each request to the best-suited sub-system.

### How It Works

```mermaid
graph LR
    Q["User Query"] --> R["Router"]
    R --> C["Code Handler"]
    R --> S["Search Handler"]
    R --> M["Math Handler"]
    R --> F["Fallback / General Handler"]
```

The router itself can be:
1. **LLM-based** — ask the model to classify the intent and return a category
2. **Embedding-based** — compute similarity between the query and category descriptions
3. **Rule-based** — keyword matching or regex patterns (fast but brittle)

### LLM Router Example

```python
ROUTER_PROMPT = \"\"\"Classify the user query into exactly one category:
- "search": needs web information
- "code": needs code generation or debugging
- "math": needs calculation
- "general": everything else

Query: {query}
Category:\"\"\"

def route(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap model suffices
        messages=[{"role": "user",
                   "content": ROUTER_PROMPT.format(query=query)}],
        max_tokens=10,
    )
    category = response.choices[0].message.content.strip().lower()
    handlers = {
        "search": handle_search,
        "code": handle_code,
        "math": handle_math,
    }
    return handlers.get(category, handle_general)(query)
```

### Fallback Strategies

Robust routing requires fallback logic:

- **Confidence thresholds** — if the router's classification confidence is below a threshold, route to a general handler
- **Multi-label routing** — allow queries to match multiple handlers and merge results
- **Cascading fallback** — try the primary handler first; if it fails or returns low-quality output, fall back to a more general handler

### When to Use Routing

Routing shines when you have **distinct, well-defined task categories** with different optimal strategies. For an Ask-the-Web agent, you might route factual queries to search, opinion queries to a discussion handler, and code questions to a code-execution tool. The key benefit is that each handler can have its own optimized prompt, tool set, and model — leading to better results than a one-size-fits-all approach.

<!-- checkpoint: I understand how routing classifies inputs and dispatches them to specialized handlers -->
""",
                },
                {
                    "title": "Parallelization",
                    "content": """## Parallelization

Parallelization runs **multiple LLM calls concurrently** to improve throughput and reduce latency. Two main sub-patterns exist:

| Pattern | Description |
|---|---|
| **Fan-out / Fan-in** | Split input into parts, process in parallel, merge results |
| **Voting / Ensembling** | Run the same prompt N times, aggregate (majority vote, best-of-N) |

### Fan-Out / Fan-In

This is the LLM equivalent of **map-reduce**. You break a large task into independent chunks, process each in parallel, then combine the outputs.

```mermaid
graph LR
    Input --> C1["LLM: Chunk 1"] --> Agg["Aggregator"]
    Input --> C2["LLM: Chunk 2"] --> Agg
    Input --> C3["LLM: Chunk 3"] --> Agg
    Agg --> Output
```

### Asyncio Example

```python
import asyncio
from openai import AsyncOpenAI

aclient = AsyncOpenAI()

async def analyze_chunk(chunk: str) -> str:
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user",
                   "content": f"Summarize this text:\\n{chunk}"}],
    )
    return response.choices[0].message.content

async def parallel_summarize(text: str) -> str:
    # Fan-out: split into chunks
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

    # Process all chunks concurrently
    summaries = await asyncio.gather(
        *[analyze_chunk(c) for c in chunks]
    )

    # Fan-in: merge summaries
    combined = "\\n".join(summaries)
    final = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Combine these summaries into one "
                              f"coherent summary:\\n{combined}"}],
    )
    return final.choices[0].message.content
```

### Voting / Ensembling

For tasks where correctness matters more than speed, run the same prompt multiple times with `temperature > 0` and take the majority answer. This is the basis of **self-consistency prompting**.

### Practical Considerations

- **Rate limits** — respect API rate limits; use semaphores to cap concurrency
- **Cost** — parallel calls multiply your token spend; use cheaper models where possible
- **Error handling** — use `asyncio.gather(return_exceptions=True)` so one failure doesn't kill the batch
- **Ordering** — results may arrive out of order; track indices if order matters

Parallelization is essential for an Ask-the-Web agent: you can search multiple sources simultaneously, analyze multiple pages concurrently, and even run multiple candidate answers in parallel before selecting the best one.
""",
                },
                {
                    "title": "Reflection",
                    "content": """## Reflection

Reflection is the pattern where an LLM **critiques its own output** and iteratively refines it. This creates a self-improvement loop that can significantly boost quality on complex tasks like writing, reasoning, and code generation.

### The Reflection Loop

```mermaid
graph LR
    Generate --> Critique --> Revise --> Critique2["Critique"] --> Revise2["Revise"] --> Accept
    Accept -->|"stop when quality threshold met"| Generate
```

### How It Works

1. **Generate** — produce an initial output
2. **Critique** — a second LLM call (or the same model with a critic prompt) evaluates the output against quality criteria
3. **Revise** — feed the critique back and ask for an improved version
4. **Terminate** — stop when the critic is satisfied, or after a maximum number of iterations

### Implementation

```python
def reflect(query: str, max_rounds: int = 3) -> str:
    # Initial generation
    draft = generate(query)

    for _ in range(max_rounds):
        # Self-critique
        critique = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user",
                 "content": f"Critique this answer for accuracy, "
                            f"completeness, and clarity. If it's good "
                            f"enough, respond with 'APPROVED'.\\n\\n"
                            f"Question: {query}\\n"
                            f"Answer: {draft}"}
            ],
        ).choices[0].message.content

        if "APPROVED" in critique:
            return draft

        # Revise based on critique
        draft = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user",
                 "content": f"Improve this answer based on the "
                            f"critique.\\nAnswer: {draft}\\n"
                            f"Critique: {critique}"}
            ],
        ).choices[0].message.content

    return draft
```

### Variants

- **LLM-as-Judge** — use a stronger model to critique a weaker model's output
- **Rubric-based** — provide explicit scoring criteria (1-5 scale on relevance, accuracy, etc.)
- **Tool-augmented** — run unit tests or search to verify factual claims
- **Constitutional AI** — critique against a set of principles or rules

### When to Use Reflection

Reflection is most valuable when: (1) output quality is critical, (2) you have clear evaluation criteria, and (3) the task benefits from iteration (writing, code, analysis). It's less useful for simple factual lookups. For your Ask-the-Web agent, reflection can verify that synthesized answers are actually supported by the retrieved sources — catching hallucinations before they reach the user.
""",
                },
                {
                    "title": "Orchestration-worker",
                    "content": """## Orchestration-Worker Pattern

The orchestration-worker pattern uses a **manager agent** that decomposes a complex task into subtasks, delegates each subtask to specialized **worker agents**, and then aggregates the results.

### Architecture

```mermaid
graph TD
    O1["ORCHESTRATOR<br/>(Manager Agent)"] --> W1["Worker 1<br/>(Search)"]
    O1 --> W2["Worker 2<br/>(Analyze)"]
    O1 --> W3["Worker 3<br/>(Compile)"]
    W1 --> O2["ORCHESTRATOR<br/>(Aggregation)"]
    W2 --> O2
    W3 --> O2
```

### How It Works

1. **Decomposition** — the orchestrator analyzes the task and breaks it into a plan of subtasks
2. **Delegation** — each subtask is assigned to a worker (which may be an LLM with a specialized prompt, a tool call, or another agent)
3. **Execution** — workers execute independently (potentially in parallel)
4. **Aggregation** — the orchestrator collects results, resolves conflicts, and synthesizes a final answer

### Example: Research Orchestrator

```python
def orchestrate_research(question: str) -> str:
    # Step 1: Decompose into sub-questions
    plan = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Break this question into 3 independent "
                              f"research sub-questions:\\n{question}"}]
    ).choices[0].message.content

    sub_questions = parse_sub_questions(plan)

    # Step 2: Delegate to workers (parallel)
    import asyncio
    results = asyncio.run(asyncio.gather(
        *[research_worker(sq) for sq in sub_questions]
    ))

    # Step 3: Aggregate
    combined = "\\n---\\n".join(results)
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Synthesize these research findings into "
                              f"a comprehensive answer:\\n{combined}"}]
    ).choices[0].message.content
    return final
```

### Key Design Decisions

- **Static vs. dynamic plans** — static plans are predefined; dynamic plans are generated by the orchestrator per query
- **Worker specialization** — each worker can have its own prompt, model, and tool access
- **Error handling** — if a worker fails, the orchestrator can retry, reassign, or proceed without that result
- **Iteration** — the orchestrator may review the aggregated result and trigger additional worker rounds

This pattern is the backbone of production AI agents like coding assistants and research tools. Your Ask-the-Web agent will use a lightweight version: decompose the question, search in parallel, and synthesize.
""",
                },
            ],
        },
        {
            "title": "Tools",
            "content": """## Tools for LLM Agents

**Tools** are the bridge between an LLM's reasoning capabilities and the real world. Without tools, an LLM is limited to its training data and can only generate text. With tools, it can search the web, execute code, query databases, call APIs, and manipulate files.

### Why Agents Need Tools

LLMs have fundamental limitations that tools address:

| Limitation | Tool Solution |
|---|---|
| Knowledge cutoff | Web search, API calls |
| Can't do math reliably | Calculator, code execution |
| Can't access private data | Database queries, file I/O |
| Can't take actions | API calls, shell commands |
| Hallucinate facts | Retrieval tools (RAG) |

### Types of Tools

1. **Information Retrieval** — web search (Google, Bing, Tavily), vector database search, knowledge base lookup
2. **Code Execution** — Python interpreter, JavaScript runtime, shell commands
3. **API Integration** — REST/GraphQL calls to external services (weather, stocks, email)
4. **File I/O** — read/write files, parse PDFs, process images
5. **Communication** — send emails, post to Slack, create GitHub issues

### The Tool-Use Loop

```mermaid
graph LR
    A["User Query"] --> B["LLM decides to use a tool"]
    B --> C["Tool call is formatted"]
    C --> D["Tool is executed"]
    D --> E["Result returned to LLM"]
    E --> F["LLM reasons over result"]
    F -->|"repeat if needed"| B
    F --> G["Final answer"]
```

### Tool Design Principles

- **Clear names and descriptions** — the LLM chooses tools based on their descriptions, so make them precise and unambiguous
- **Minimal parameters** — fewer parameters mean fewer chances for the LLM to make mistakes
- **Informative error messages** — when a tool fails, return a clear error so the LLM can recover
- **Idempotent when possible** — safe to retry on failure

For your Ask-the-Web agent, the primary tool is **web search**, but you'll likely also need a **page reader** (to fetch and parse full web pages) and possibly a **calculator** for data-heavy queries. The key insight is that the LLM decides *when* and *how* to use each tool based on the user's query — this decision-making is what makes it an agent rather than a pipeline.

<!-- checkpoint: I understand why agents need tools and the tool-use loop -->
""",
            "children": [
                {
                    "title": "Tool calling",
                    "content": """## Tool Calling

Tool calling (also called function calling) is the mechanism by which an LLM **requests execution of an external function**. The LLM doesn't execute the tool itself — it outputs a structured request that your application intercepts, executes, and returns.

### OpenAI Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What happened today?"}],
    tools=tools,
    tool_choice="auto",  # LLM decides whether to call a tool
)

# Check if the model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(tool_call.function.name)       # "web_search"
    print(tool_call.function.arguments)  # '{"query": "today news"}'
```

### Anthropic Tool Use

Anthropic uses a similar but slightly different format:

```python
response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {  # Note: "input_schema" not "parameters"
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }],
    messages=[{"role": "user", "content": "Latest news?"}]
)
```

### Key Concepts

- **tool_choice** — controls whether the model must call a tool (`"required"`), may call one (`"auto"`), or must not (`"none"`)
- **Parallel tool calls** — models can request multiple tool calls in a single response
- **Structured outputs** — both OpenAI and Anthropic support strict JSON schema adherence for reliable parsing
- The LLM produces a **JSON arguments blob**; your code parses it and invokes the actual function
- Always validate tool call arguments before execution — LLMs can produce malformed or unexpected values

<!-- checkpoint: I understand how tool calling works with OpenAI and Anthropic APIs -->
""",
                },
                {
                    "title": "Tool formatting",
                    "content": """## Tool Formatting

Tool formatting is the art of **describing your tools to the LLM** so it can reliably choose and invoke them. The quality of your tool descriptions directly impacts how well the agent performs — vague descriptions lead to wrong tool choices and malformed arguments.

### JSON Schema Foundation

Both OpenAI and Anthropic use **JSON Schema** to define tool parameters:

```json
{
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query. Be specific and use keywords."
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results (1-20)",
            "enum": [5, 10, 20]
        },
        "date_range": {
            "type": "string",
            "description": "Filter by date. ISO 8601 format.",
            "enum": ["past_day", "past_week", "past_month", "any"]
        }
    },
    "required": ["query"],
    "additionalProperties": false
}
```

### OpenAI vs Anthropic Format Comparison

| Aspect | OpenAI | Anthropic |
|---|---|---|
| Schema key | `parameters` | `input_schema` |
| Wrapper | `{"type": "function", "function": {...}}` | `{...}` (flat) |
| Tool choice | `tool_choice: "auto"` | `tool_choice: {"type": "auto"}` |
| Parallel calls | Supported natively | Supported natively |
| Strict mode | `"strict": true` | Not needed (strict by default) |

### Writing Effective Descriptions

**Bad:**
```json
{"name": "search", "description": "Searches stuff"}
```

**Good:**
```json
{
    "name": "web_search",
    "description": "Search the web using Google. Returns titles, URLs, and snippets for each result. Use for current events, factual questions, or when the user asks about something after your knowledge cutoff. Do NOT use for calculations or code execution."
}
```

### Best Practices

- **Be explicit about when NOT to use** a tool — this prevents false positives
- **Include examples** in descriptions for ambiguous parameters
- **Use `enum`** for constrained values instead of free-text
- **Mark parameters as `required`** only if truly necessary — optional params with defaults reduce LLM errors
- **Keep parameter count low** — 3-5 parameters max per tool; split complex tools into simpler ones
- **Use descriptive parameter names** — `search_query` is better than `q`

Good formatting is the cheapest way to improve agent reliability. Before adding complex retry logic, first ensure your tool descriptions are clear and complete.
""",
                },
                {
                    "title": "Tool execution",
                    "content": """## Tool Execution

Tool execution is the runtime layer that actually **runs the tool** after the LLM requests it. This layer must handle sandboxing, error recovery, timeouts, and security — because the LLM is effectively generating code or commands that your system executes.

### The Execution Pipeline

```mermaid
graph LR
    A["LLM Tool Call"] --> B["Validate Args"]
    B --> C["Sandbox Check"]
    C --> D["Execute"]
    D --> E["Handle Errors"]
    E --> F["Return Result"]
```

### Sandboxing

Never execute LLM-generated tool calls in an unsandboxed environment. Options include:

| Sandbox | Best For | Latency |
|---|---|---|
| **Docker containers** | Code execution, shell commands | ~1-2s startup |
| **E2B sandboxes** | Cloud-hosted code execution | ~500ms (pre-warmed) |
| **AWS Lambda** | Stateless API calls | ~100ms (warm) |
| **gVisor / Firecracker** | High-security isolation | ~200ms |

### Error Handling

```python
import json
from typing import Any

def execute_tool(name: str, arguments: str) -> dict[str, Any]:
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON arguments"}

    tool_fn = TOOL_REGISTRY.get(name)
    if not tool_fn:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = tool_fn(**args)
        return {"success": True, "result": result}
    except TimeoutError:
        return {"error": "Tool execution timed out after 30s"}
    except Exception as e:
        return {"error": f"Tool failed: {type(e).__name__}: {str(e)}"}
```

### Retry Logic

Not all failures should be retried. Use a policy:

- **Retryable** — network timeouts, rate limits (429), transient server errors (503)
- **Non-retryable** — invalid arguments, authentication errors, resource not found
- **Exponential backoff** — wait 1s, 2s, 4s between retries; cap at 3 attempts

### Timeout Management

Every tool call needs a timeout. Without one, a hanging API call blocks your entire agent loop.

```python
import asyncio

async def safe_execute(tool_fn, args, timeout=30):
    try:
        return await asyncio.wait_for(tool_fn(**args), timeout=timeout)
    except asyncio.TimeoutError:
        return {"error": f"Timed out after {timeout}s"}
```

### Security Considerations

- **Allowlist tools** — only register tools the agent should have access to
- **Validate arguments** — check types, ranges, and patterns before execution
- **Rate-limit tool calls** — prevent runaway loops (cap at 20 tool calls per query)
- **Log everything** — record every tool call and result for auditing and debugging
""",
                },
                {
                    "title": "MCP",
                    "content": """## Model Context Protocol (MCP)

The **Model Context Protocol** (MCP), created by Anthropic, is an open standard that provides a **universal interface between AI applications and external tools/data sources**. Think of it as "USB-C for AI agents" — a single protocol that replaces dozens of custom integrations.

### The Problem MCP Solves

Without MCP, every AI application must build custom integrations for every tool:

```
Before MCP:  N apps × M tools = N×M custom integrations
With MCP:    N apps + M tools = N+M standardized connections
```

### Architecture

MCP follows a **client-server model**:

```mermaid
graph TD
    subgraph "AI Application"
        MC1["MCP Client"]
        MC2["MCP Client"]
    end
    MC1 --> S1["MCP Server<br/>(GitHub)"]
    MC2 --> S2["MCP Server<br/>(Database)"]
```

- **MCP Host** — the AI application (Claude Desktop, VS Code, your custom app)
- **MCP Client** — protocol handler inside the host; one client per server
- **MCP Server** — exposes tools, resources, and prompts over the protocol

### Transport Mechanisms

| Transport | Use Case | How It Works |
|---|---|---|
| **stdio** | Local servers | Server runs as a subprocess; communicates via stdin/stdout |
| **SSE (HTTP)** | Remote servers | Server exposes an HTTP endpoint; uses Server-Sent Events for streaming |
| **Streamable HTTP** | Modern remote | Newer transport replacing SSE; supports bidirectional streaming |

### What MCP Servers Expose

1. **Tools** — functions the LLM can call (e.g., `search_files`, `run_query`)
2. **Resources** — data the LLM can read (e.g., file contents, database schemas)
3. **Prompts** — reusable prompt templates for common tasks

### Tool Discovery

MCP clients discover available tools dynamically by calling `tools/list` on the server. This means your agent doesn't need hardcoded tool definitions — it can discover what's available at runtime.

### Benefits

- **Standardization** — build once, connect everywhere
- **Security** — servers control access; data stays local when using stdio
- **Composability** — connect multiple MCP servers to one agent
- **Community ecosystem** — growing library of pre-built MCP servers (GitHub, Slack, databases, file systems)

MCP is rapidly becoming the standard for tool integration in production AI systems.
""",
                },
            ],
        },
        {
            "title": "Multi-Step Agents",
            "content": """## Multi-Step Agents

Multi-step agents go beyond single-turn question answering by **planning and executing a sequence of actions** to solve complex problems. While a simple LLM call gives you one response, a multi-step agent reasons about what to do, takes an action, observes the result, and repeats until the task is complete.

### Why Single-Turn Isn't Enough

Consider the query: *"Compare the GDP growth of India and China over the last 5 years."* A single LLM call would rely entirely on training data (potentially outdated) and guesswork. A multi-step agent would:

1. Search for India's GDP data (2021-2025)
2. Search for China's GDP data (2021-2025)
3. Extract numerical values from search results
4. Calculate growth rates
5. Synthesize a comparison

Each step builds on the results of previous steps, and the agent dynamically decides what to do next based on what it finds.

### The Planning + Execution Paradigm

Multi-step agents operate in a loop:

```
while task_not_complete:
    thought = reason_about_current_state()
    action  = choose_next_action(thought)
    result  = execute_action(action)
    update_state(result)
```

### Key Agent Frameworks

| Framework | Core Idea |
|---|---|
| **ReACT** | Interleave reasoning (Thought) with acting (Action/Observation) |
| **Reflexion** | Add self-reflection after failures to improve future attempts |
| **ReWOO** | Plan all reasoning steps first, then execute without interleaving |
| **Tree Search** | Explore multiple solution paths, backtrack when stuck |

### Trade-Offs

- **Accuracy vs. Cost** — more steps = more API calls = higher cost
- **Autonomy vs. Control** — fully autonomous agents can go off-track; adding checkpoints helps
- **Speed vs. Thoroughness** — thorough agents take longer but produce better results

The choice of framework depends on your task. For your Ask-the-Web agent, **ReACT** is the most natural fit: the agent reasons about what to search, searches, observes results, and decides whether it has enough information to answer or needs another search.
""",
            "children": [
                {
                    "title": "ReACT",
                    "content": """## ReACT: Reasoning + Acting

**ReACT** (Reason + Act) is the foundational multi-step agent framework. It interleaves **reasoning traces** (Thought) with **tool use** (Action) and **results** (Observation) in a loop until the task is complete.

### The Loop

```mermaid
graph LR
    T["Thought"] --> A["Action"] --> O["Observation"] --> T2["Thought"] --> A2["Action"] --> O2["Observation"] --> FA["Final Answer"]
```

The key innovation is that the **Thought** step is explicit — the LLM verbalizes its reasoning before deciding what action to take. This makes the agent's decision-making transparent and debuggable.

### Full Example Trace

**User Query:** *"Who won the most recent Nobel Prize in Physics and what was their key contribution?"*

```
Thought 1: I need to find who won the most recent Nobel Prize in Physics.
            Let me search for this information.
Action 1:  web_search("Nobel Prize Physics 2025 winner")
Observation 1: The 2025 Nobel Prize in Physics was awarded to
               John Hopfield and Geoffrey Hinton for foundational
               discoveries in machine learning with artificial
               neural networks.

Thought 2: I found the winners. Now I need more detail on their
            specific contributions. Let me search for more info.
Action 2:  web_search("Hopfield Hinton Nobel Physics contribution details")
Observation 2: Hopfield created an associative memory network
               (Hopfield network). Hinton invented the Boltzmann
               machine and backpropagation improvements...

Thought 3: I now have enough information to provide a comprehensive
            answer. Let me synthesize.
Action 3:  final_answer("The 2025 Nobel Prize in Physics was awarded
            to John Hopfield and Geoffrey Hinton for...")
```

### Implementation Skeleton

```python
REACT_PROMPT = \"\"\"Answer the user's question using the available tools.
For each step, output:
Thought: <your reasoning>
Action: <tool_name>(<arguments>)

When you have the final answer:
Thought: I have enough information.
Action: final_answer(<your answer>)\"\"\"

def react_loop(query: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": query},
    ]
    for _ in range(max_steps):
        response = llm_call(messages)
        action, args = parse_action(response)
        if action == "final_answer":
            return args
        observation = execute_tool(action, args)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user",
                         "content": f"Observation: {observation}"})
    return "Max steps reached without answer."
```

### Strengths and Weaknesses

- ✅ Transparent reasoning — every decision is logged
- ✅ Flexible — works with any set of tools
- ❌ Token-expensive — the full trace grows with each step
- ❌ Can loop — without good stopping criteria, agents may repeat actions
""",
                },
                {
                    "title": "Reflexion",
                    "content": """## Reflexion

**Reflexion** extends multi-step agents with **self-reflection after failure**. When an agent produces an incorrect or suboptimal result, it generates a verbal critique of what went wrong and stores it in an **episodic memory**. On the next attempt, the agent consults this memory to avoid repeating the same mistakes.

### How Reflexion Works

```mermaid
graph LR
    Actor["Actor<br/>(ReACT agent)"] --> Evaluator["Evaluator<br/>(Pass/Fail)"]
    Evaluator --> Reflect["Reflect<br/>(Self-Critique)"]
    Reflect --> Memory["Episodic Memory"]
    Memory --> Actor
```

### The Three Components

1. **Actor** — a ReACT-style agent that attempts the task
2. **Evaluator** — determines if the attempt succeeded (binary pass/fail, unit tests, or LLM-as-judge)
3. **Self-Reflection** — analyzes the failed attempt and generates a natural language lesson

### Example

**Task:** Write a function that returns the second-largest element in a list.

**Attempt 1:** Agent writes code, evaluator runs unit tests → 2 of 5 tests fail.

**Reflection:**
> "My function failed on lists with duplicate elements. I used `set()` to remove
> duplicates before sorting, which is incorrect when the second-largest should
> account for duplicates. I should sort the original list and find the second
> distinct value by iterating."

**Attempt 2:** Agent consults the reflection memory, writes corrected code → all tests pass.

### Episodic Memory

The reflections are stored as a list of strings and prepended to the agent's prompt on subsequent attempts:

```python
class ReflexionAgent:
    def __init__(self):
        self.memory: list[str] = []

    def run(self, task: str, max_trials: int = 3) -> str:
        for trial in range(max_trials):
            # Include past reflections
            context = "\\n".join(self.memory) if self.memory else "None"
            result = self.actor.run(task, past_reflections=context)

            if self.evaluator.check(result):
                return result  # Success!

            # Generate reflection on failure
            reflection = self.reflect(task, result)
            self.memory.append(reflection)

        return result  # Return best attempt
```

### Verbal Reinforcement Learning

Reflexion can be seen as **reinforcement learning with verbal feedback** instead of scalar rewards. The reflection serves as a rich, informative signal that helps the agent improve much faster than a simple reward would. This is particularly powerful for coding tasks, where specific error messages and test failures provide concrete feedback for the reflection step.

### Limitations

- Requires a reliable evaluator (unit tests, structured checks)
- Memory grows with each attempt — may need summarization
- Not all tasks have clear success criteria for the evaluator
""",
                },
                {
                    "title": "ReWOO",
                    "content": """## ReWOO: Reasoning WithOut Observation

**ReWOO** (Reasoning WithOut Observation) takes a fundamentally different approach from ReACT. Instead of interleaving reasoning and acting step-by-step, ReWOO **plans all steps upfront** and then executes them in sequence. The LLM reasons once to create a complete plan, without seeing intermediate results during planning.

### ReWOO vs ReACT

```mermaid
graph LR
    subgraph ReACT
        T1["Think"] --> A1["Act"] --> O1["Observe"] --> T2["Think"] --> A2["Act"] --> O2["Observe"] --> Ans1["Answer"]
    end
```

```mermaid
graph LR
    subgraph ReWOO
        P["Plan ALL steps"] --> E["Execute ALL steps"] --> S["Synthesize Answer"]
    end
```

### The Three Stages

1. **Planner** — generates a full plan with tool calls, using placeholder variables for results
2. **Worker** — executes each planned tool call in order, filling in the actual results
3. **Solver** — synthesizes the final answer from all collected evidence

### Example Plan

**Query:** *"Is the population of France larger than the population of Germany?"*

```
Plan:
#E1 = web_search("population of France 2025")
#E2 = web_search("population of Germany 2025")
#E3 = calculate("compare #E1 and #E2")

Execution:
#E1 → "France population: ~68.4 million"
#E2 → "Germany population: ~84.5 million"
#E3 → "Germany (84.5M) > France (68.4M)"

Solver: "No, the population of France (~68.4 million) is smaller
         than Germany (~84.5 million)."
```

### Implementation

```python
def rewoo(query: str) -> str:
    # Stage 1: Plan
    plan = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Create a step-by-step plan to answer: "
                              f"{query}\\nUse #E1, #E2, etc. as "
                              f"placeholders for tool results."}]
    ).choices[0].message.content

    steps = parse_plan(plan)  # Extract tool calls and dependencies

    # Stage 2: Execute
    evidence = {}
    for step_id, tool_name, args in steps:
        # Replace placeholder references with actual results
        resolved_args = resolve_placeholders(args, evidence)
        evidence[step_id] = execute_tool(tool_name, resolved_args)

    # Stage 3: Solve
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Answer '{query}' using this "
                              f"evidence:\\n{format_evidence(evidence)}"}]
    ).choices[0].message.content
```

### Advantages Over ReACT

- **Token efficiency** — the LLM is called only twice (plan + solve) instead of once per step, drastically reducing token usage
- **Parallelizable** — independent steps in the plan can be executed concurrently
- **Lower latency** — fewer LLM calls means faster end-to-end execution

### Disadvantages

- **No adaptation** — the plan can't adjust based on intermediate results (e.g., if a search returns nothing useful, the agent can't reformulate)
- **Brittle on complex tasks** — tasks requiring dynamic decision-making suffer without the observe-then-think loop
- Best suited for **well-structured tasks** where the steps are predictable
""",
                },
                {
                    "title": "Tree search",
                    "content": """## Tree Search for LLM Agents

Tree search applies **search algorithms** (like Monte Carlo Tree Search) to the agent decision space. Instead of committing to a single path of actions, the agent explores **multiple possible action sequences** as branches of a tree and selects the most promising path.

### Why Tree Search?

ReACT-style agents follow a single path. If they make a wrong decision early, they're stuck. Tree search addresses this by:

1. **Branching** — generating multiple candidate actions at each step
2. **Evaluating** — scoring each branch with a value function
3. **Selecting** — pursuing the most promising branch
4. **Backtracking** — returning to earlier states when a path fails

### Tree Structure

```mermaid
graph TD
    Root["Root: User Query"] --> SA["Search A"]
    Root --> SB["Search B"]
    Root --> SC["Search C"]
    SA --> FU["Follow-up"]
    SA --> DE["Dead End"]
    SB --> EX["Extract"]
    SC --> RE["Refine"]
    EX --> AnsA["Answer<br/>score=0.9 ✓"]
    RE --> AnsB["Answer<br/>score=0.7"]
```

### MCTS for LLMs (Tree of Thoughts)

The **Tree of Thoughts** (ToT) framework applies MCTS principles:

```python
def tree_search(question: str, breadth: int = 3,
                depth: int = 3) -> str:
    root = Node(state=question, parent=None)
    best_answer = None
    best_score = 0

    def expand(node, current_depth):
        nonlocal best_answer, best_score
        if current_depth >= depth:
            return

        # Generate multiple candidate next steps
        candidates = generate_candidates(node.state, n=breadth)

        for candidate in candidates:
            child = Node(state=candidate, parent=node)

            # Evaluate this branch
            score = evaluate(child.state, question)

            if score > best_score:
                best_score = score
                best_answer = child.state

            # Only expand promising branches (pruning)
            if score > 0.5:
                expand(child, current_depth + 1)

    expand(root, 0)
    return best_answer
```

### Evaluation Functions

The quality of tree search depends on the **evaluation function** at each node:

- **LLM-as-judge** — ask a model to rate the partial solution (0-1)
- **Heuristic scoring** — count relevant keywords, check factual consistency
- **Self-consistency** — generate multiple completions and check agreement
- **Unit tests** (for code) — run tests to get a concrete pass/fail signal

### Trade-Offs

- ✅ **Explores diverse solutions** — finds answers that linear agents miss
- ✅ **Handles ambiguity** — multiple interpretations can be pursued in parallel
- ❌ **Expensive** — branching factor × depth = many LLM calls
- ❌ **Complex implementation** — state management, pruning, and backtracking add engineering overhead

Tree search is most valuable for **hard reasoning tasks** (math, code, puzzles) where the search space is large but evaluable. For straightforward web search tasks, ReACT is usually sufficient and more cost-effective.
""",
                },
            ],
        },
        {
            "title": "Multi-Agent Systems",
            "content": """## Multi-Agent Systems

Multi-agent systems (MAS) involve **multiple AI agents working together** — each with its own role, tools, and potentially its own LLM — to accomplish tasks that are too complex for a single agent. The agents communicate, coordinate, and sometimes debate to produce better outcomes.

### Communication Patterns

Multi-agent systems use different topologies for agent communication:

```mermaid
graph LR
    subgraph "1. Sequential"
        A1["A"] --> B1["B"] --> C1["C"] --> D1["D"]
    end
```

```mermaid
graph TD
    subgraph "2. Hierarchical"
        M["Manager"] --> W1["W1"]
        M --> W2["W2"]
        M --> W3["W3"]
    end
```

```mermaid
graph LR
    subgraph "3. Peer-to-Peer"
        A3["A"] <--> B3["B"]
        A3 <--> C3["C"]
        B3 <--> D3["D"]
        C3 <--> D3
    end
```

```mermaid
graph LR
    subgraph "4. Broadcast"
        A4["A"] --> B4["B"]
        A4 --> C4["C"]
        A4 --> D4["D"]
    end
```

```mermaid
graph LR
    subgraph "5. Blackboard"
        A5["A"] --> SS["Shared State"]
        B5["B"] --> SS
        C5["C"] --> SS
        SS --> R["Reader"]
    end
```

### Key Design Decisions

1. **Agent specialization** — give each agent a focused role (researcher, writer, critic, coder) with a tailored system prompt and tool set
2. **Communication protocol** — how agents share information (direct messages, shared memory, structured handoffs)
3. **Coordination strategy** — who decides what happens next (central orchestrator vs. emergent behavior)
4. **Termination condition** — when does the system stop? (consensus, quality threshold, max rounds)

### Popular Frameworks

| Framework | Architecture | Key Feature |
|---|---|---|
| **AutoGen** (Microsoft) | Conversational agents | Agents chat in group conversations |
| **CrewAI** | Role-based crews | Agents with roles, goals, and backstories |
| **LangGraph** | Graph-based workflows | Agents as nodes in a state machine |
| **Swarm** (OpenAI) | Lightweight handoffs | Agents transfer control via function returns |

### When to Use Multi-Agent

Multi-agent is warranted when:
- The task has **naturally distinct roles** (researcher + writer + editor)
- You need **adversarial validation** (generator vs. critic)
- **Different tools/models** are optimal for different subtasks
- You want **parallel specialized processing**

Avoid multi-agent when a single well-prompted agent with tools can handle the task — the coordination overhead of multi-agent systems can outweigh the benefits for simpler problems.
""",
            "children": [
                {
                    "title": "Challenges",
                    "content": """## Challenges in Multi-Agent Systems

Building multi-agent systems introduces **coordination complexity** that doesn't exist in single-agent architectures. Understanding these challenges is essential for designing robust systems.

### 1. Coordination Overhead

Every message between agents costs tokens and latency. In a system with N agents that all communicate, the number of potential message pairs grows as O(N²). This leads to:

- **Token explosion** — agents passing context back and forth can consume enormous token budgets
- **Latency accumulation** — sequential agent interactions add up; a 5-agent pipeline with 2s per agent = 10s minimum
- **Diminishing returns** — adding more agents often adds more overhead than value

### 2. Consistency and State Management

When multiple agents work on the same task, keeping a **consistent view of the world** is hard:

```
Agent A searches → finds X is true
Agent B searches → finds X is false (different source)
Agent C receives both → confused, contradicts itself
```

Solutions include:
- **Single source of truth** — use a shared state object that agents read/write to
- **Structured handoffs** — each agent explicitly passes its findings in a structured format
- **Conflict resolution** — an arbiter agent resolves disagreements

### 3. Deadlocks and Infinite Loops

Agents can get stuck in conversational loops:

```
Critic: "This needs more detail on X."
Writer: (adds detail on X)
Critic: "Good, but now Y needs more detail."
Writer: (adds detail on Y)
Critic: "Now X needs updating because of Y..."
→ Infinite loop
```

**Prevention strategies:**
- Set a **maximum number of rounds** for any agent interaction
- Use **convergence detection** — if the output stops changing meaningfully, terminate
- Implement **escalation** — after N rounds, a supervisor agent makes the final call

### 4. Error Propagation

In a pipeline of agents, errors compound. If Agent A produces a slightly wrong output, Agent B builds on that error, and by Agent C the result may be completely wrong.

**Mitigation:**
- Add **validation gates** between agents (same as in prompt chaining)
- Use **redundancy** — have multiple agents do the same task and compare
- Implement **rollback** — if a downstream agent detects an error, restart from an earlier stage

### 5. Debugging and Observability

Multi-agent systems are notoriously hard to debug. When the final output is wrong, which agent caused the problem? **Logging and tracing** are essential:

- Log every agent's input, output, and tool calls
- Use **trace IDs** to track a query through the entire system
- Visualize agent communication graphs to spot bottlenecks
- Implement **LangSmith** or similar observability tools from day one
""",
                },
                {
                    "title": "Use-cases",
                    "content": """## Multi-Agent Use Cases

Multi-agent systems shine in scenarios where **diverse expertise, adversarial validation, or parallel processing** are required. Here are the most impactful real-world applications.

### 1. Code Review Agents

A multi-agent code review system uses specialized agents for different concerns:

```mermaid
graph TD
    Sec["Security Agent"] --> Agg["Aggregator<br/>(Dedupes & ranks)"]
    Sty["Style Agent"] --> Agg
    Perf["Perf Agent"] --> Agg
    Logic["Logic Agent"] --> Agg
```

Each agent reviews the same diff but focuses on its specialty. The aggregator deduplicates overlapping concerns and prioritizes the most critical issues.

### 2. Debate / Adversarial Agents

Two or more agents argue opposing positions to surface the strongest arguments:

- **Pro Agent** — argues *for* a position
- **Con Agent** — argues *against*
- **Judge Agent** — evaluates arguments and declares a winner or synthesis

This pattern is used in **Constitutional AI** training and in decision-support systems where exploring both sides prevents premature conclusions.

### 3. Research Teams

A team of agents collaborates on deep research:

- **Planner** — breaks the research question into sub-questions
- **Searchers** (multiple) — each investigates a sub-question using web search
- **Analyst** — synthesizes findings, identifies gaps
- **Writer** — produces the final report
- **Editor** — reviews for accuracy and coherence

This mirrors how human research teams work and can produce reports that rival junior analyst output.

### 4. Simulation and Role-Playing

Agents simulate real-world interactions:

- **Customer service training** — one agent plays the customer, another the support rep; a coach agent provides feedback
- **Market simulation** — buyer and seller agents negotiate to model market dynamics
- **Red team / Blue team** — adversarial agents probe a system for vulnerabilities while defender agents try to block them

### 5. Software Engineering Teams

Full development workflows with agent teams:

- **PM Agent** — writes requirements and user stories from a brief
- **Developer Agent** — writes code based on requirements
- **Tester Agent** — writes and runs tests
- **Reviewer Agent** — reviews code for issues

Frameworks like **ChatDev** and **MetaGPT** implement this pattern, demonstrating that agent teams can produce working software from a single-sentence description.
""",
                },
                {
                    "title": "A2A protocol",
                    "content": """## A2A: Agent-to-Agent Protocol

**A2A** (Agent-to-Agent) is Google's open protocol for **inter-agent communication**. While MCP connects agents to *tools*, A2A connects agents to *other agents* — enabling different AI systems to discover, communicate with, and delegate tasks to each other.

### A2A vs MCP

```mermaid
graph LR
    MCP["MCP: Agent"] <--> Tool["Tool/Data Source"]
    A2A["A2A: Agent"] <--> Agent2["Agent"]
```

| Aspect | MCP (Anthropic) | A2A (Google) |
|---|---|---|
| **Purpose** | Connect agents to tools | Connect agents to agents |
| **Discovery** | `tools/list` endpoint | Agent Cards (JSON metadata) |
| **Communication** | Tool calls + results | Task lifecycle (messages) |
| **State** | Stateless tool calls | Stateful task management |
| **Complementary?** | Yes — A2A agents can use MCP tools internally |

### Core Concepts

#### 1. Agent Cards

Every A2A agent publishes an **Agent Card** — a JSON document at `/.well-known/agent.json` that describes what the agent can do:

```json
{
    "name": "WebResearchAgent",
    "description": "Searches the web and summarizes findings",
    "url": "https://research-agent.example.com",
    "capabilities": {
        "streaming": true,
        "pushNotifications": true
    },
    "skills": [
        {
            "id": "web-search",
            "name": "Web Search",
            "description": "Search and summarize web content"
        }
    ]
}
```

#### 2. Task Lifecycle

A2A uses a **task-based communication model**:

```mermaid
sequenceDiagram
    participant C as Client Agent
    participant R as Remote Agent
    C->>R: Create Task
    R-->>C: Status: submitted
    R-->>C: Status: working
    R-->>C: (streaming artifacts)
    R-->>C: Status: completed
    R-->>C: Final Artifacts
```

Task states: `submitted` → `working` → `completed` (or `failed`, `canceled`)

#### 3. Artifacts and Messages

- **Messages** — conversational turns between agents (with `role: "user"` or `role: "agent"`)
- **Artifacts** — the actual outputs (generated text, files, structured data)
- Each artifact has `parts` — text, file, or structured data

### Why A2A Matters

A2A enables an **ecosystem of specialized agents** that can discover and hire each other. Your web-research agent could delegate a coding task to a code-execution agent, which could delegate a deployment task to a DevOps agent — all through standardized protocols. Combined with MCP for tool access, A2A completes the vision of fully interoperable AI systems.
""",
                },
            ],
        },
        {
            "title": "Agent Evaluation",
            "content": """## Agent Evaluation

Evaluating agents is fundamentally harder than evaluating simple LLM outputs because agents take **multi-step actions with real-world side effects**. You need to measure not just *what* the agent produced, but *how* it got there — was it efficient? Did it use the right tools? Did it recover from errors?

### Key Metrics

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Success Rate** | Did the agent complete the task correctly? | correct_completions / total_tasks |
| **Step Efficiency** | How many steps did it take? | actual_steps / optimal_steps |
| **Tool Accuracy** | Did it choose the right tools? | correct_tool_calls / total_tool_calls |
| **Cost Efficiency** | How many tokens/dollars per task? | total_tokens / successful_tasks |
| **Recovery Rate** | Can it recover from errors? | recovered_errors / total_errors |
| **Latency** | End-to-end time to completion | wall_clock_time per task |

### Established Benchmarks

#### SWE-bench

Tests an agent's ability to **fix real GitHub issues**. The agent receives a GitHub issue description and must produce a code patch that passes the repository's test suite. SWE-bench Verified contains ~500 human-validated instances.

```
Input:  GitHub issue + repository codebase
Output: Git patch
Eval:   Run repository tests → pass/fail
```

#### WebArena

Tests **web navigation and interaction**. The agent must complete tasks on real websites (Reddit, GitLab, shopping sites) by clicking, typing, and navigating — evaluated by checking the final state of the website.

#### GAIA (General AI Assistants)

Tests **multi-step reasoning with tool use**. Questions require combining web search, calculation, file reading, and reasoning. GAIA has three difficulty levels with human-verified answers.

```
Example: "What was the mass of the heaviest animal ever recorded
          in kg, rounded to the nearest thousand?"
Requires: search → find the animal → find the record → convert units
```

### Building Your Own Evaluation

For your Ask-the-Web agent, create a custom evaluation suite:

```python
test_cases = [
    {
        "query": "What is the current population of Tokyo?",
        "criteria": {
            "has_number": True,
            "source_cited": True,
            "recency": "within_1_year",
        },
        "max_steps": 5,
    },
]

def evaluate_agent(agent, test_cases):
    results = []
    for tc in test_cases:
        trace = agent.run(tc["query"])
        results.append({
            "success": check_criteria(trace.answer, tc["criteria"]),
            "steps": len(trace.steps),
            "tools_used": trace.tool_calls,
            "tokens": trace.total_tokens,
            "latency_s": trace.elapsed_seconds,
        })
    return aggregate_metrics(results)
```

### Best Practices

- **Test on diverse queries** — factual, comparative, temporal, multi-hop
- **Measure failure modes** — categorize *why* the agent fails (wrong tool, bad search query, hallucination, loop)
- **Track regressions** — run your eval suite on every agent change to catch regressions
- **Human evaluation** — automated metrics miss nuance; periodically have humans rate agent outputs for quality, helpfulness, and factual accuracy
- **Cost tracking** — monitor cost per query in production; set budget caps per agent run
""",
            "children": [],
        },
    ],
}

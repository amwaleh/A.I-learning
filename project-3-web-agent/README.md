# Project 3: Build an "Ask-the-Web" Agent — Learning Material

## Table of Contents

1. [Agents Overview](#1-agents-overview)
2. [Workflows](#2-workflows)
3. [Tools](#3-tools)
4. [Multi-Step Agents](#4-multi-step-agents)
5. [Multi-Agent Systems](#5-multi-agent-systems)
6. [Agent Evaluation](#6-agent-evaluation)
7. [Glossary](#7-glossary)

---

## 1. Agents Overview

### What is an Agent?

An **agent** is a program that uses an LLM (Large Language Model) to decide what actions to take. Unlike a simple chatbot that just answers questions, an agent can *do things* — search the web, run code, call APIs, and more.

Think of it like this:

```mermaid
graph LR
    subgraph "Plain LLM"
        Q1[Question] --> A1[Answer]
    end
    subgraph Agent
        Q2[Question] --> T1[Think] --> Act1[Act] --> T2[Think] --> Act2[Act] --> T3[Think] --> A2[Answer]
    end
```

### LLM vs. Agent vs. Agentic System

| Concept | What It Is | Example |
|---------|-----------|---------|
| **LLM** | A model that generates text | GPT-4, Claude, Llama |
| **Agent** | LLM + tools + decision loop | A bot that searches the web then answers |
| **Agentic System** | Multiple agents working together | One agent researches, another writes, a third reviews |

```mermaid
graph TD
    subgraph "LLM (Brain only)"
        M1[Model] -. "I think the answer is..." .-> M1
    end
    subgraph "Agent (Brain + Hands)"
        M2[Model] -->|calls| T2[Tools]
        T2 -->|results| M2
        M2 -.->|"Loop until done"| M2
    end
    subgraph "Agentic System (Team)"
        A1["Agent 1\nResearcher"] --- A2["Agent 2\nWriter"] --- A3["Agent 3\nReviewer"]
    end
```

### Levels of Agency

Agency exists on a spectrum — from fully scripted workflows to fully autonomous agents:

```mermaid
graph LR
    subgraph Low Agency
        W["Workflow\n(fixed steps)"]
        R["Router\n(picks a path)"]
    end
    subgraph High Agency
        RE["ReACT\n(thinks + acts)"]
        AU["Autonomous\n(plans and\nself-corrects)"]
    end
    W --> R --> RE --> AU
    HC["Human controls\nevery step"] -.-> W
    AU -.-> AC["Agent controls\nits own actions"]
```

**Key Insight:** More agency = more flexibility, but also more risk of errors. Start simple and add agency only when needed.

---

## 2. Workflows

Workflows are **pre-defined patterns** for how an LLM processes information. The human designs the flow; the LLM executes each step.

### 2.1 Prompt Chaining

Run one LLM call, then feed its output into the next call.

```mermaid
graph LR
    S1["Step 1\nResearch"] -->|"Output 1\nfeeds into Step 2"| S2["Step 2\nOutline"] -->|"Output 2\nfeeds into Step 3"| S3["Step 3\nWrite\n(Final result)"]
```

**Example: Summarize then Translate**

```python
import openai

client = openai.OpenAI()

# Step 1: Summarize
response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Summarize this article: {article_text}"}]
)
summary = response1.choices[0].message.content

# Step 2: Translate the summary
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Translate to French: {summary}"}]
)
french_summary = response2.choices[0].message.content
```

**When to use:** When each step has a clear input/output and later steps depend on earlier ones.

---

### 2.2 Routing

The LLM decides which path to take based on the input.

```mermaid
graph TD
    R["Router\n(classifies input)"]
    R --> A["Path A:\nSimple Q"]
    R --> B["Path B:\nResearch"]
    R --> C["Path C:\nCreative"]
```

**Example:**

```python
def route_question(question):
    """Use LLM to classify the question and pick a handler."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Classify this question into one category:
            - 'factual' (needs web search)
            - 'creative' (needs creative writing)
            - 'code' (needs code generation)

            Question: {question}
            Reply with just the category name."""
        }]
    )
    category = response.choices[0].message.content.strip()

    if category == "factual":
        return handle_factual(question)
    elif category == "creative":
        return handle_creative(question)
    else:
        return handle_code(question)
```

**When to use:** When different types of input need different processing strategies.

---

### 2.3 Parallelization

Run multiple LLM calls at the same time for speed or better results.

#### Sectioning (split work)

```mermaid
graph TD
    I[Input] --> W1["Worker 1\n(Part A)"]
    I --> W2["Worker 2\n(Part B)"]
    I --> W3["Worker 3\n(Part C)"]
    W1 --> C[Combine]
    W2 --> C
    W3 --> C
```

**Example:** Analyze a document from 3 angles simultaneously:

```python
import asyncio
import openai

client = openai.AsyncOpenAI()

async def analyze_parallel(document):
    """Analyze a document from multiple angles in parallel."""
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"List key facts in: {document}"}]
        ),
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"What is the sentiment of: {document}"}]
        ),
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize in 2 sentences: {document}"}]
        ),
    ]
    results = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in results]
```

#### Voting (multiple opinions)

Run the same prompt multiple times and take the majority answer:

```python
async def vote_on_answer(question, num_votes=3):
    """Ask the same question multiple times and take majority answer."""
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}]
        )
        for _ in range(num_votes)
    ]
    results = await asyncio.gather(*tasks)
    answers = [r.choices[0].message.content for r in results]
    # Pick most common answer
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]
```

---

### 2.4 Reflection

The LLM reviews and improves its own output.

```mermaid
graph LR
    G["Generate\n(draft)"] --> C["Critique\n(review)"] --> I["Improve\n(revise)"]
    I -->|"Loop until good"| C
```

**Example:**

```python
def generate_with_reflection(task, max_iterations=3):
    """Generate content, then iteratively improve it."""
    # Step 1: Generate initial draft
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Write: {task}"}]
    )
    draft = response.choices[0].message.content

    for i in range(max_iterations):
        # Step 2: Critique
        critique_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""
            Critique this writing. List specific problems:
            {draft}
            If it's good enough, say 'APPROVED'.
            """}]
        )
        critique = critique_response.choices[0].message.content

        if "APPROVED" in critique:
            break

        # Step 3: Improve based on critique
        improve_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""
            Improve this writing based on the feedback:

            Original: {draft}
            Feedback: {critique}
            """}]
        )
        draft = improve_response.choices[0].message.content

    return draft
```

---

### 2.5 Orchestration-Worker

A central "orchestrator" LLM breaks down work and delegates to workers.

```mermaid
graph TD
    O1["Orchestrator\n(plans & assigns subtasks)"] --> W1["Worker 1\n(subtask)"]
    O1 --> W2["Worker 2\n(subtask)"]
    O1 --> W3["Worker 3\n(subtask)"]
    W1 --> O2["Orchestrator\n(combines results)"]
    W2 --> O2
    W3 --> O2
```

**Example:**

```python
def orchestrator(complex_question):
    """Break a complex question into subtasks and combine results."""
    # Plan subtasks
    plan_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""
        Break this question into 2-3 simpler sub-questions:
        {complex_question}
        Return as a numbered list.
        """}]
    )
    subtasks = plan_response.choices[0].message.content

    # Execute each subtask (workers)
    worker_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""
        Answer each sub-question:
        {subtasks}
        """}]
    )
    answers = worker_response.choices[0].message.content

    # Combine into final answer
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""
        Combine these answers into one coherent response:
        Original question: {complex_question}
        Sub-answers: {answers}
        """}]
    )
    return final_response.choices[0].message.content
```

---

## 3. Tools

### 3.1 What is Tool Calling?

Tool calling lets an LLM **request the use of external functions**. The LLM doesn't run the tool itself — it tells your program *which* tool to call and *what arguments* to pass.

```mermaid
sequenceDiagram
    participant User
    participant LLM
    participant Code as Your Code
    participant Tool as get_weather()

    User->>LLM: "What's the weather in Tokyo?"
    LLM->>Code: tool: "get_weather", args: {city: "Tokyo"}
    Code->>Tool: get_weather("Tokyo")
    Tool-->>Code: "72°F, sunny"
    Code->>LLM: Result: "72°F, sunny"
    LLM->>User: "It's 72°F and sunny in Tokyo!"
```

**Key Insight:** The LLM decides WHAT to do. Your code does the actual work.

---

### 3.2 Tool Formatting

Tools are described to the LLM using a specific JSON format:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

**Parts of a tool definition:**

| Field | Purpose | Example |
|-------|---------|---------|
| `name` | Identifier for the tool | `"search_web"` |
| `description` | Tells the LLM when to use it | `"Search the web for..."` |
| `parameters` | What inputs the tool needs | `query`, `num_results` |
| `required` | Which parameters are mandatory | `["query"]` |

---

### 3.3 Tool Execution

Here's the complete flow for executing a tool call:

```python
import json

def run_conversation(user_message, tools):
    """Complete tool-calling loop."""
    messages = [{"role": "user", "content": user_message}]

    # Step 1: Send message to LLM with available tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    # Step 2: Check if LLM wants to use a tool
    if message.tool_calls:
        messages.append(message)  # Add assistant's response to history

        # Step 3: Execute each tool call
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Call the actual function
            result = call_function(function_name, arguments)

            # Step 4: Send result back to LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # Step 5: Get final response
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        return final_response.choices[0].message.content

    return message.content


def call_function(name, args):
    """Dispatch to the actual function implementation."""
    if name == "search_web":
        return search_web(args["query"], args.get("num_results", 5))
    elif name == "get_weather":
        return get_weather(args["city"])
    else:
        return f"Unknown function: {name}"
```

---

### 3.4 MCP (Model Context Protocol)

MCP is a **standardized way** to connect LLMs to external tools and data sources. Think of it as a "USB port" for AI — any tool that speaks MCP can plug into any LLM that supports it.

```mermaid
graph TD
    subgraph "WITHOUT MCP"
        LA1["LLM A"] -->|custom code| T1a["Tool 1"]
        LA2["LLM A"] -->|custom code| T2a["Tool 2"]
        LB1["LLM B"] -->|"custom code (rewrite!)"| T1b["Tool 1"]
        LB2["LLM B"] -->|"custom code (rewrite!)"| T2b["Tool 2"]
    end
    subgraph "WITH MCP"
        LA["LLM A"] -->|MCP| S1["MCP Server\n(Tool 1)"]
        LB["LLM B"] -->|MCP| S1
        LA3["LLM A"] -->|MCP| S2["MCP Server\n(Tool 2)"]
        LB3["LLM B"] -->|MCP| S2
    end
```

**MCP Components:**

| Component | Role | Analogy |
|-----------|------|---------|
| **MCP Host** | The application (IDE, chatbot) | Your computer |
| **MCP Client** | Connects host to servers | USB cable |
| **MCP Server** | Provides tools/resources | USB device |

**Why MCP Matters:**
- Tools are reusable across different LLMs
- Standard format means less custom code
- Growing ecosystem of pre-built MCP servers

---

## 4. Multi-Step Agents

Multi-step agents can **plan and execute multiple actions** autonomously. They decide what to do next based on what they've learned so far.

### 4.1 Planning Autonomy

```mermaid
graph LR
    FP["Fixed Plan\n'Do steps 1, 2, 3'\n(Workflows)"] --- AP["Adaptive Plan\n'Plan first,\nadjust as you go'\n(ReWOO)"] --- NP["No Plan\n'Figure it out\nas you go'\n(ReACT)"]
```

---

### 4.2 ReACT (Reasoning + Acting)

ReACT is the most popular agent pattern. The agent alternates between **thinking** (reasoning) and **doing** (acting).

```mermaid
flowchart TD
    T1["THOUGHT\n'I need to find out about X'"] --> A1["ACTION\nsearch_web('topic X')"]
    A1 --> O1["OBSERVATION\n'Results: ...'"]
    O1 --> T2["THOUGHT\n'Now I know X, but I need Y'"]
    T2 --> A2["ACTION\nsearch_web('topic Y')"]
    A2 --> O2["OBSERVATION\n'Results: ...'"]
    O2 --> AN["ANSWER\n'Based on X and Y, here's...'"]
```

**Simple ReACT Example:**

```python
def react_agent(question, tools, max_steps=5):
    """A simple ReACT agent that thinks and acts in a loop."""
    messages = [
        {"role": "system", "content": """You are a helpful assistant.
        Think step by step. Use tools when you need information.
        When you have enough info, provide a final answer."""},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message

        # If no tool calls, agent is done — return answer
        if not message.tool_calls:
            return message.content

        # Execute tool calls (Action)
        messages.append(message)
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        # Loop back — LLM will Observe and Reason next

    return "Agent reached max steps without a final answer."
```

---

### 4.3 Reflexion

Reflexion adds **self-evaluation** to the agent loop. After acting, the agent asks itself: "Did that work? What should I do differently?"

```mermaid
flowchart TD
    Act --> Evaluate
    Evaluate -->|"Was my answer correct?"| Reflect
    Reflect -->|"What went wrong?"| Improve
    Improve --> Act
```

```python
def reflexion_agent(question, max_attempts=3):
    """Agent that reflects on and improves its answers."""
    memory = []  # Store past reflections

    for attempt in range(max_attempts):
        # Generate answer (with past reflections as context)
        answer = generate_answer(question, memory)

        # Evaluate the answer
        evaluation = evaluate_answer(question, answer)

        if evaluation["is_correct"]:
            return answer

        # Reflect on what went wrong
        reflection = reflect(question, answer, evaluation["feedback"])
        memory.append(reflection)

    return answer  # Return best attempt
```

---

### 4.4 ReWOO (Reasoning Without Observation)

ReWOO plans ALL actions upfront before executing any of them. This saves tokens and time.

```mermaid
graph TD
    subgraph "Phase 1: PLAN (all at once)"
        P1["Step 1: Search for X"]
        P2["Step 2: Search for Y"]
        P3["Step 3: Compare X and Y"]
    end
    subgraph "Phase 2: EXECUTE (run all tools)"
        R1["Result 1: ..."]
        R2["Result 2: ..."]
    end
    subgraph "Phase 3: SOLVE (combine results)"
        F["Final answer based on\nall collected information"]
    end
    P1 --> R1
    P2 --> R2
    P3 --> F
    R1 --> F
    R2 --> F
```

**Comparison: ReACT vs ReWOO**

| Feature | ReACT | ReWOO |
|---------|-------|-------|
| Planning | Step by step | All upfront |
| LLM calls | Many (think-act-observe loop) | Few (plan, then solve) |
| Adaptability | High (adjusts after each step) | Lower (fixed plan) |
| Token usage | Higher | Lower |
| Best for | Complex, uncertain tasks | Straightforward multi-step tasks |

---

### 4.5 Tree Search for Agents

When there are many possible paths, agents can explore multiple options like a tree:

```mermaid
graph TD
    S[Start] --> A["Option A\nScore: 3"]
    S --> B["Option B\nScore: 8"]
    S --> C["Option C\nScore: 5"]
    B --> B1["B1\nScore: 6"]
    B --> B2["B2 ✓\nScore: 9"]
    B --> B3["B3\nScore: 4"]
    B2 -->|Best path!| Done((Done))
```

**Concepts:**
- **Branching:** Try multiple approaches at each step
- **Scoring:** Rate how promising each branch is
- **Pruning:** Abandon low-scoring branches early
- **Backtracking:** If stuck, go back and try a different path

This is useful for complex reasoning tasks like math proofs or code generation.

---

## 5. Multi-Agent Systems

### 5.1 What Are Multi-Agent Systems?

Multiple specialized agents collaborating on a task, each with their own role and tools.

```mermaid
graph TD
    U["User Question:\n'Write a blog post about AI trends'"]
    U --> CO["Coordinator\n(assigns tasks, manages flow)"]
    CO --> R["Researcher\n(searches the web)"]
    CO --> W["Writer\n(writes content)"]
    CO --> E["Editor\n(reviews and improves)"]
```

### 5.2 Challenges

| Challenge | Description |
|-----------|-------------|
| **Communication** | Agents need a clear protocol to share information |
| **Coordination** | Who does what? How to avoid duplicate work? |
| **Error propagation** | One agent's mistake can cascade to others |
| **Debugging** | Hard to trace what went wrong in a multi-agent system |
| **Cost** | More agents = more LLM calls = higher cost |

### 5.3 Use Cases

- **Software Development:** Coder + Tester + Reviewer
- **Research:** Searcher + Analyzer + Writer
- **Customer Support:** Router + Specialist agents for different topics
- **Data Processing:** Extractor + Transformer + Validator

### 5.4 A2A Protocol (Agent-to-Agent)

A2A is a **standard protocol** (by Google) for agents to communicate with each other, similar to how MCP standardizes tool access.

```mermaid
graph LR
    A["Agent A\n(any vendor)"] <-->|"A2A Protocol\nStandard messages\nfor task delegation"| B["Agent B\n(any vendor)"]
```

**Key concepts:**
- **Agent Card:** describes what an agent can do
- **Task:** unit of work with lifecycle (submitted → working → completed)
- **Message/Part:** structured communication

**MCP vs A2A:**
- **MCP** = LLM ↔ Tool (like calling a function)
- **A2A** = Agent ↔ Agent (like collaborating with a colleague)

---

## 6. Agent Evaluation

How do you know if your agent works well? Here are key metrics:

### Evaluation Dimensions

```mermaid
graph TD
    subgraph "EVALUATION FRAMEWORK"
        Acc["Accuracy\nDid it get the\nright answer?"]
        Eff["Efficiency\nHow many\nsteps/calls?"]
        Rel["Reliability\nDoes it work\nconsistently?"]
        Saf["Safety\nDid it avoid\nharmful actions?"]
        Cos["Cost\nHow many\ntokens/$ used?"]
        Lat["Latency\nHow long\ndid it take?"]
    end
```

### Evaluation Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| **Human evaluation** | People judge agent outputs | Quality, safety |
| **LLM-as-judge** | Another LLM grades the output | Scale, consistency |
| **Automated metrics** | Compare to known answers | Factual accuracy |
| **Trajectory analysis** | Review the steps taken | Efficiency, debugging |

### Simple Evaluation Example

```python
def evaluate_agent(agent, test_cases):
    """Run agent on test cases and measure performance."""
    results = []

    for test in test_cases:
        import time
        start = time.time()

        answer = agent(test["question"])

        elapsed = time.time() - start

        # Check correctness (simple keyword match)
        is_correct = any(
            keyword.lower() in answer.lower()
            for keyword in test["expected_keywords"]
        )

        results.append({
            "question": test["question"],
            "correct": is_correct,
            "time_seconds": elapsed,
            "answer_length": len(answer)
        })

    # Summary
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)

    print(f"Accuracy: {accuracy:.0%}")
    print(f"Avg time: {avg_time:.1f}s")

    return results
```

---

## 7. Glossary

| Term | Definition |
|------|-----------|
| **Agent** | A program that uses an LLM to decide what actions to take |
| **Agentic System** | Multiple agents working together on a task |
| **LLM** | Large Language Model — the AI brain (e.g., GPT-4, Claude) |
| **Tool** | A function the LLM can ask to be called (search, calculator, etc.) |
| **Tool Calling** | The LLM's ability to request function execution |
| **MCP** | Model Context Protocol — standard for connecting LLMs to tools |
| **A2A** | Agent-to-Agent protocol — standard for agent communication |
| **ReACT** | Reasoning + Acting — agent thinks then acts in a loop |
| **Reflexion** | Agent pattern that includes self-evaluation and improvement |
| **ReWOO** | Reasoning Without Observation — plan all steps before executing |
| **Prompt Chaining** | Output of one LLM call feeds into the next |
| **Routing** | LLM classifies input and picks the right handler |
| **Parallelization** | Running multiple LLM calls simultaneously |
| **Orchestrator** | A central agent that delegates work to other agents |
| **Observation** | The result returned after an agent uses a tool |
| **Token** | A chunk of text (~4 characters) — LLM pricing unit |
| **Context Window** | Maximum amount of text an LLM can process at once |
| **Trajectory** | The sequence of steps an agent took to reach an answer |
| **Grounding** | Connecting LLM responses to real-world data (web, docs) |

---

## Summary Diagram: How It All Fits Together

```mermaid
flowchart TD
    Q[USER QUESTION] --> Think1["Think: What tool do I need?"]
    Think1 --> Act["Act: Call tool (via Tool Calling)"]
    Act --> Observe["Observe: Read the result"]
    Observe --> Think2{"Think: Do I need more info?"}
    Think2 -->|Yes| Act
    Think2 -->|No| Answer["FINAL ANSWER\n(with citations)"]
```

---

## Next Steps

Ready to build? Head to [PROJECT.md](./PROJECT.md) to build your own "Ask-the-Web" agent step by step!

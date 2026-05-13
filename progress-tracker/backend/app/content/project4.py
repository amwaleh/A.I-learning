PROJECT_4 = {
    "number": 4,
    "title": "Build Deep Research Capability",
    "description": "Build Deep Research Capability",
    "topics": [
        {
            "title": "Inference-time Techniques",
            "content": """\
## Inference-time Techniques

Inference-time techniques represent a paradigm shift in how we extract intelligence from large language models. Rather than relying solely on a single forward pass, these methods allocate **additional compute at test time** to dramatically improve reasoning quality.

### The Core Insight

The key discovery is that **scaling compute at inference** can be as powerful as scaling model size during training. A smaller model given more time to "think" can outperform a larger model that answers immediately. This mirrors human cognition — difficult problems benefit from deliberation.

### Taxonomy of Approaches

Inference-time techniques fall into two broad families:

| Family | Mechanism | Examples |
|--------|-----------|----------|
| **Parallel** | Generate multiple candidates, select the best | Self-consistency, Best-of-N |
| **Sequential** | Iteratively refine a single reasoning chain | CoT, critique-revise loops |
| **Structured search** | Explore a tree/graph of reasoning paths | Tree of Thoughts, MCTS |

### Why This Matters

Traditional scaling laws focused on pre-training compute (Chinchilla scaling). Inference-time scaling introduces a **second axis of improvement**:

```
Total capability = f(training compute) + g(inference compute)
```

This means a well-trained base model can be "boosted" at deployment time by spending more FLOPs per query. For high-stakes tasks (math proofs, code generation, medical reasoning), the cost of extra inference compute is justified by the quality gains.

### Practical Considerations

- Not all tasks benefit equally — factual recall gains little from extra thinking
- Diminishing returns exist — there is a compute-optimal frontier
- Latency trade-offs must be managed for real-time applications
- Verification is critical — more compute helps only when you can distinguish good from bad reasoning

These techniques form the foundation of modern reasoning systems like OpenAI's o1/o3 and DeepSeek-R1.
""",
            "children": [
                {
                    "title": "Inference-time scaling",
                    "content": """\
## Inference-time Scaling

Inference-time scaling studies how allocating more compute during generation improves model performance, analogous to how training-time scaling laws govern pre-training.

### Compute-Optimal Inference

The central question is: **given a fixed inference budget, what is the best way to spend it?** Research from Google DeepMind and others has shown that the optimal strategy depends on the difficulty of the problem:

```
Easy questions  →  Single pass is sufficient
Medium questions →  Chain-of-thought helps
Hard questions  →  Search + verification is optimal
```

### Scaling Laws for Test-Time Compute

Snell et al. (2024) established **test-time compute scaling laws** showing predictable relationships:

| Strategy | Scaling Behavior | Best For |
|----------|-----------------|----------|
| Majority voting (N samples) | log-linear improvement | Well-calibrated models |
| Best-of-N with verifier | Stronger than voting | When verifiers are available |
| Sequential revision | Sublinear gains | Iterative refinement tasks |
| Tree search + PRM | Near-linear in compute | Math/logic problems |

### When to Allocate More Thinking

The optimal allocation follows a **difficulty-aware** strategy:

1. **Estimate problem difficulty** — Use model confidence, token entropy, or a classifier
2. **Route easy problems** — Single-pass generation (low latency, low cost)
3. **Route hard problems** — Multi-step search with verification (high cost, high accuracy)

```python
def adaptive_inference(prompt, model, verifier):
    # Quick attempt
    answer = model.generate(prompt)
    confidence = verifier.score(prompt, answer)

    if confidence > 0.9:
        return answer  # Easy — single pass

    # Hard — invest more compute
    candidates = [model.generate(prompt) for _ in range(32)]
    scores = [verifier.score(prompt, c) for c in candidates]
    return candidates[argmax(scores)]
```

### Key Takeaway

Inference-time scaling is **complementary** to training-time scaling. The compute-optimal frontier considers both axes, and modern systems dynamically adjust inference budgets per query to maximize quality under cost constraints.
""",
                },
                {
                    "title": "CoT prompting",
                    "content": """\
## Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting instructs models to produce intermediate reasoning steps before arriving at a final answer. Introduced by Wei et al. (2022), CoT dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks.

### Variants of CoT

**Zero-shot CoT** — Simply append "Let's think step by step" to any prompt:

```
Q: If a train travels 120 miles in 2 hours, what is its speed?
A: Let's think step by step.
   - Distance = 120 miles
   - Time = 2 hours
   - Speed = Distance / Time = 120 / 2 = 60 mph
   The answer is 60 mph.
```

**Few-shot CoT** — Provide exemplars with reasoning chains:

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each.
   How many does he have now?
A: Roger starts with 5 balls. He buys 2 × 3 = 6 balls.
   Total = 5 + 6 = 11. The answer is 11.

Q: [Your actual question here]
A:
```

**Structured CoT** — Enforce specific reasoning formats:

```
## Given:  [extract known facts]
## Goal:   [state what to find]
## Steps:  [numbered reasoning]
## Answer: [final result]
```

### Effectiveness by Task Type

| Task Type | CoT Benefit | Why |
|-----------|------------|-----|
| Multi-step math | ★★★★★ | Breaks computation into steps |
| Logical deduction | ★★★★☆ | Makes premises explicit |
| Commonsense QA | ★★★☆☆ | Surfaces implicit knowledge |
| Factual recall | ★☆☆☆☆ | No reasoning needed |
| Creative writing | ★★☆☆☆ | Marginal planning benefit |

### Why CoT Works

CoT works because it converts **System 1** (fast, intuitive) processing into **System 2** (slow, deliberate) processing. By forcing intermediate tokens, the model conditions each step on prior reasoning rather than jumping directly to an answer. This is especially critical for tasks requiring **compositional reasoning** where the answer depends on combining multiple facts.

### Limitations

- Increases token usage and latency
- Can produce plausible-sounding but incorrect reasoning (faithfulness problem)
- Small models (<10B params) gain less from CoT
""",
                },
                {
                    "title": "Parallel sampling",
                    "content": """\
## Parallel Sampling

Parallel sampling generates **multiple independent solutions** to the same problem, then selects the best one. This is one of the simplest yet most effective inference-time scaling strategies.

### Core Approaches

**Majority Voting (Self-Consistency)** — Sample N responses, extract answers, take the majority vote:

```mermaid
graph LR
    P[Prompt] --> S1["Sample 1 → Answer: 42"]
    P --> S2["Sample 2 → Answer: 42"]
    P --> S3["Sample 3 → Answer: 37"]
    P --> S4["Sample 4 → Answer: 42"]
    S1 --> M["Majority: 42 ✓"]
    S2 --> M
    S3 --> M
    S4 --> M
```

Introduced by Wang et al. (2022), self-consistency decoding marginalizes over reasoning paths to find the most consistent answer. It requires no additional training — only multiple samples from the same model with temperature > 0.

**Best-of-N with Verifier** — Generate N candidates, score each with a verifier, return the highest-scored:

```python
def best_of_n(prompt, model, verifier, n=16):
    candidates = [model.generate(prompt, temperature=0.7) for _ in range(n)]
    scores = [verifier.score(prompt, c) for c in candidates]
    return candidates[scores.index(max(scores))]
```

**Reward-Weighted Sampling** — Weight candidates by their reward scores rather than taking the argmax, producing a soft selection.

### Scaling Properties

| N (samples) | Relative Accuracy Gain | Cost |
|-------------|----------------------|------|
| 1 | Baseline | 1× |
| 4 | +5-10% | 4× |
| 16 | +10-18% | 16× |
| 64 | +15-22% | 64× |
| 256 | +18-25% | 256× |

Gains follow a **log-linear** pattern — doubling N gives roughly constant improvement. This means diminishing returns, but the simplicity of the approach makes it practical.

### When to Use Parallel Sampling

- Problems with **verifiable answers** (math, code, constrained generation)
- When latency allows batched generation
- When a reliable scoring function exists
- As a baseline before trying more complex search methods

Parallel sampling is embarrassingly parallelizable, making it efficient on GPU clusters despite the raw compute cost.
""",
                },
                {
                    "title": "Sequential sampling",
                    "content": """\
## Sequential Sampling

Sequential sampling uses **iterative refinement** where each generation step builds upon and improves the previous attempt. Unlike parallel sampling which generates independent candidates, sequential methods create a dependency chain.

### Core Patterns

**Critique-Then-Revise** — Generate an initial answer, critique it, then produce a revised answer:

```mermaid
graph LR
    A["Generate\nDraft v1"] --> B["Critique\nFind flaws"]
    B --> C["Revise\nDraft v2"]
    C -->|Repeat| B
```

```python
def critique_revise(prompt, model, max_rounds=3):
    draft = model.generate(f"Solve: {prompt}")
    for _ in range(max_rounds):
        critique = model.generate(
            f"Find errors in this solution:\n{draft}"
        )
        if "no errors" in critique.lower():
            break
        draft = model.generate(
            f"Original: {draft}\nCritique: {critique}\n"
            f"Provide a corrected solution:"
        )
    return draft
```

**Progressive Deepening** — Start with a high-level plan, then elaborate each section iteratively:

1. Generate outline → 2. Expand key sections → 3. Fill details → 4. Verify consistency

**Refinement with External Feedback** — Use tool outputs (code execution, search results) to inform the next iteration:

```
Generate code → Execute → Read errors → Fix → Execute → Pass ✓
```

### Advantages Over Parallel Sampling

| Aspect | Parallel | Sequential |
|--------|----------|------------|
| Independence | Fully independent | Each step informed by prior |
| Diversity | High diversity | Focused refinement |
| Error correction | No self-correction | Explicit error fixing |
| Compute efficiency | Wasteful if early samples good | Adaptive effort |

### When Sequential Shines

Sequential sampling excels when:
- The model can **reliably identify its own errors** (strong self-critique)
- Problems have **clear intermediate checkpoints** (compilable code, parseable math)
- External **verification tools** are available (interpreters, theorem provers)
- The initial draft is **close to correct** and needs targeted fixes

### Limitations

- Prone to **anchoring** — the model may fixate on the initial approach
- Self-critique can be unreliable, especially for subtle errors
- Latency scales linearly with refinement rounds
- Risk of "refinement collapse" where quality degrades after too many rounds
""",
                },
                {
                    "title": "Tree of Thoughts",
                    "content": """\
## Tree of Thoughts (ToT)

Tree of Thoughts (Yao et al., 2023) extends chain-of-thought by exploring **multiple branching reasoning paths** organized as a tree. Each node represents a partial solution, and the system uses search algorithms to find the best path.

### Tree Structure

```mermaid
graph TD
    P["Problem"] --> A1["Step A₁"]
    P --> A2["Step A₂"]
    P --> A3["Step A₃"]
    A1 --> B1["Step B₁"]
    A1 --> B2["Step B₂ ✗"]
    A2 --> B3["Step B₃"]
    A3 --> B4["Step B₄ ✗"]
    B1 --> C1["Step C₁"]
    B3 --> C2["Step C₂"]
    C1 --> Ans1["Answer₁"]
    C2 --> Ans2["Answer₂ ← Best"]
```

Each branching point generates **multiple candidate next steps**. An evaluation function scores partial solutions, and unpromising branches (✗) are pruned.

### Search Strategies

**BFS (Breadth-First Search)** — Explore all nodes at depth d before moving to d+1. Good when the branching factor is small and evaluation is reliable at early stages:

```python
def tot_bfs(problem, model, evaluator, breadth=3, depth=3):
    current_level = [problem]
    for d in range(depth):
        candidates = []
        for node in current_level:
            # Generate multiple next steps
            steps = [model.propose_step(node) for _ in range(breadth)]
            candidates.extend([(node + s, evaluator.score(node + s)) for s in steps])
        # Keep top-k candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        current_level = [c[0] for c in candidates[:breadth]]
    return current_level[0]
```

**DFS (Depth-First Search)** — Explore one branch deeply, backtrack on failure. Efficient when solutions are deep and evaluation is costly.

### Evaluation Functions

The evaluator at each node is critical. Common approaches:

| Method | Description | Reliability |
|--------|-------------|-------------|
| LLM self-evaluation | "Rate this partial solution 1-10" | Moderate |
| Value function | Trained classifier predicting success | High |
| Heuristic scoring | Domain-specific rules | Variable |
| Simulation | Complete the solution, check answer | High but costly |

### Applications

ToT excels at tasks requiring **exploration and backtracking**: creative writing, puzzle-solving (Game of 24, crosswords), planning problems, and multi-step mathematical proofs. It consistently outperforms standard CoT on problems where the first reasoning path is often wrong and backtracking is necessary.

### Cost Considerations

ToT is compute-intensive — a tree with breadth b and depth d requires O(bᵈ) evaluations. Practical implementations use aggressive pruning and caching to manage costs.
""",
                },
                {
                    "title": "Search against verifier",
                    "content": """\
## Search Against Verifier

Search against verifier combines **systematic exploration** of reasoning paths with **learned verification models** that score each step or final answer. This is the most powerful inference-time technique, underlying systems like AlphaProof and OpenAI's o1.

### Process Reward Models (PRM)

PRMs assign a score to **each intermediate step** in a reasoning chain, enabling fine-grained feedback:

```
Step 1: "Let x = 5"         → PRM score: 0.95 ✓
Step 2: "Then 2x = 15"      → PRM score: 0.12 ✗  ← Error caught!
Step 3: "Therefore x² = 25" → PRM score: 0.88 ✓
```

PRMs are trained on human annotations of step-level correctness. They enable **early pruning** of incorrect reasoning paths, saving compute.

### Outcome Reward Models (ORM)

ORMs score only the **final answer**, not intermediate steps:

```
Full Solution A → ORM score: 0.92
Full Solution B → ORM score: 0.34
Full Solution C → ORM score: 0.78
```

ORMs are cheaper to train (only need final answer labels) but less efficient at search since bad paths aren't pruned early.

| Model | Granularity | Training Cost | Search Efficiency |
|-------|------------|---------------|-------------------|
| PRM | Per-step | High (step labels) | High (early pruning) |
| ORM | Final answer | Low (answer labels) | Low (full generation) |

### Monte Carlo Tree Search (MCTS) for Reasoning

Inspired by AlphaGo, MCTS adapts game-tree search to reasoning:

```python
def mcts_reasoning(problem, policy_model, verifier, simulations=100):
    root = Node(problem)
    for _ in range(simulations):
        # 1. SELECT — traverse tree using UCB1
        node = select(root)
        # 2. EXPAND — generate candidate next steps
        children = policy_model.propose_steps(node.state)
        # 3. SIMULATE — complete reasoning to get outcome
        outcomes = [simulate(child, policy_model) for child in children]
        # 4. BACKPROPAGATE — update scores up the tree
        for child, outcome in zip(children, outcomes):
            score = verifier.score(outcome)
            backpropagate(child, score)
    return best_path(root)
```

### AlphaProof-Style Approach

DeepMind's AlphaProof applied this to mathematical theorem proving: a language model proposes proof steps, a formal verifier (Lean) checks correctness, and MCTS explores the proof space. This achieved silver-medal performance at the International Mathematical Olympiad.

The key principle: **search is powerful when verification is cheap relative to generation**. Math and code naturally satisfy this — checking a proof or running a test is far easier than producing one.
""",
                },
            ],
        },
        {
            "title": "Training-time Techniques",
            "content": """\
## Training-time Techniques

Training-time techniques aim to **teach models how to reason** during the training process itself, so that improved reasoning becomes an intrinsic capability rather than requiring elaborate inference-time scaffolding.

### Why Train for Reasoning?

Inference-time techniques (CoT, ToT, search) are powerful but costly — they multiply compute per query. If a model can **internalize** these reasoning patterns during training, it can produce high-quality reasoning in a single forward pass or with minimal extra compute.

### The Two Pillars

Training-time reasoning improvements rely on two complementary approaches:

| Supervised (SFT) | Reinforcement Learning |
|---|---|
| Curated reasoning datasets | RLHF / RLAIF |
| Step-by-step demonstrations | Process rewards / Outcome rewards |
| Distillation from stronger models | Self-play / self-improve |
| | GRPO, PPO, DPO |
| | Verifier-guided training |

### The Training Pipeline

Modern reasoning models follow a multi-stage pipeline:

1. **Pre-training** — Standard language modeling on large corpora
2. **SFT on reasoning data** — Fine-tune on curated step-by-step solutions
3. **RL with verifiers** — Optimize against reward signals for correct reasoning
4. **Self-refinement** — Use the model to generate its own training data

### Key Insight: Data Quality > Data Quantity

Research consistently shows that a **small amount of high-quality reasoning data** outperforms large amounts of noisy data. For example, training on 10K expert-annotated math solutions can beat training on 1M web-scraped solutions.

### Relationship to Inference-time Techniques

Training-time and inference-time techniques are complementary. A model trained with RL on reasoning tasks will produce better chains-of-thought, which in turn benefit from tree search and verification at inference time. The best systems (o1, DeepSeek-R1) combine both approaches.
""",
            "children": [
                {
                    "title": "SFT on reasoning data",
                    "content": """\
## Supervised Fine-Tuning (SFT) on Reasoning Data

SFT on reasoning data is the most straightforward approach to teaching models to reason: collect high-quality step-by-step solutions and fine-tune the model to reproduce them.

### What Makes Good Reasoning Data?

The training examples must contain **explicit intermediate reasoning**, not just question-answer pairs:

```
❌ Poor training example:
Q: What is 23 × 17?
A: 391

✓ Good training example:
Q: What is 23 × 17?
A: Let me break this down:
   23 × 17 = 23 × (10 + 7)
           = 23 × 10 + 23 × 7
           = 230 + 161
           = 391
```

### Data Sources

| Source | Quality | Scale | Examples |
|--------|---------|-------|----------|
| Human experts | ★★★★★ | Low (1K-50K) | Math olympiad solutions |
| Teacher models | ★★★★☆ | High (100K+) | GPT-4 generated solutions |
| Filtered web data | ★★★☆☆ | Very high | StackOverflow, textbooks |
| Synthetic generation | ★★★☆☆ | Unlimited | Procedurally generated problems |

### Distillation from Stronger Models

A common and effective strategy is **distillation**: use a stronger model (e.g., GPT-4, Claude) to generate reasoning traces, then train a smaller model on these traces. DeepSeek-R1-Distill models demonstrated that distilling from a reasoning-capable teacher into Qwen and Llama base models produces surprisingly strong reasoners.

```python
# Simplified distillation pipeline
for problem in dataset:
    # Generate reasoning trace from teacher
    trace = teacher_model.generate(
        f"Solve step by step: {problem}",
        temperature=0.7
    )
    # Verify the answer is correct
    if verify_answer(trace, problem.gold_answer):
        training_data.append({"input": problem, "output": trace})

# Fine-tune student on verified traces
student_model.finetune(training_data, epochs=3, lr=2e-5)
```

### Quality vs Quantity Tradeoffs

Research findings:
- **Filtering for correctness** is essential — training on wrong reasoning is harmful
- **Diverse reasoning paths** help more than many examples of the same approach
- **Curriculum learning** (easy→hard) can improve training stability
- A dataset of 10K high-quality examples often beats 100K unfiltered ones

### Limitations of SFT Alone

SFT teaches the model to **imitate** reasoning patterns but doesn't optimize for **correctness**. The model learns to produce text that *looks like* good reasoning, which may not *be* good reasoning. This is why SFT is typically followed by RL-based training to optimize for actual problem-solving success.
""",
                },
                {
                    "title": "RL with verifier",
                    "content": """\
## Reinforcement Learning with Verifier

RL with a verifier trains the model to optimize for **actually solving problems correctly**, not just imitating solutions. The verifier provides reward signals that guide the model toward effective reasoning strategies.

### GRPO (Group Relative Policy Optimization)

GRPO, used by DeepSeek-R1, is a simplified RL algorithm that avoids the need for a separate critic model:

```python
def grpo_step(model, problems, verifier, group_size=16):
    for problem in problems:
        # Sample a group of responses
        responses = [model.generate(problem) for _ in range(group_size)]
        rewards = [verifier.score(problem, r) for r in responses]

        # Compute group-relative advantages
        mean_r = mean(rewards)
        std_r = std(rewards)
        advantages = [(r - mean_r) / std_r for r in rewards]

        # Update policy — upweight good responses
        for response, advantage in zip(responses, advantages):
            loss = -advantage * log_prob(model, response)
            loss.backward()
    model.optimizer.step()
```

The key insight: by comparing responses **within a group**, GRPO normalizes rewards without needing a learned value function.

### Process Rewards vs Outcome Rewards

| Reward Type | Signal | Granularity | Training Signal |
|------------|--------|-------------|-----------------|
| **Outcome (ORM)** | Final answer correct? | Sparse (1 per solution) | Easy to collect |
| **Process (PRM)** | Each step correct? | Dense (1 per step) | Expensive but powerful |

Process rewards provide **denser gradients**, helping the model learn *which steps* contribute to correct solutions. Outcome rewards are simpler but make credit assignment harder — the model must figure out which steps mattered.

### Online RL Training Loop

```mermaid
graph LR
    A["Sample\nproblems"] --> B["Verify\nsolutions"]
    B --> C["Update\npolicy"]
    C -->|Repeat for K iterations| A
```

### Emergent Behaviors

A remarkable finding from DeepSeek-R1: when trained with RL alone (without SFT), models spontaneously develop reasoning behaviors like **self-verification**, **backtracking**, and **exploring alternative approaches**. The model discovers these strategies because they lead to higher rewards — not because it was shown examples of them.

### Practical Considerations

- RL training is **unstable** — reward hacking, mode collapse, and catastrophic forgetting are common
- KL divergence penalties prevent the model from drifting too far from the SFT checkpoint
- Mixing RL reasoning data with general instruction data preserves broad capabilities
""",
                },
                {
                    "title": "Reward modeling",
                    "content": """\
## Reward Modeling

Reward models (RMs) are trained classifiers that predict human preferences, serving as proxy objectives for RL training. They are the critical bridge between human judgment and automated training.

### The Bradley-Terry Model

Most reward models use the **Bradley-Terry** framework for pairwise preferences:

```
P(response A preferred over B) = σ(r(A) - r(B))
```

where `r(x)` is the scalar reward assigned to response `x` and `σ` is the sigmoid function.

The training loss is:

```python
def reward_model_loss(preferred, rejected, reward_model):
    r_preferred = reward_model(preferred)
    r_rejected = reward_model(rejected)
    # Maximize margin between preferred and rejected
    loss = -log(sigmoid(r_preferred - r_rejected))
    return loss
```

### Collecting Preference Data

The pipeline for building a reward model:

1. **Generate** — Sample multiple responses per prompt from the policy model
2. **Annotate** — Human raters compare pairs (or rank multiple responses)
3. **Train** — Fit the reward model on preference data
4. **Validate** — Check agreement with held-out human judgments

| Data Collection Method | Cost | Quality | Scale |
|----------------------|------|---------|-------|
| Expert annotation | Very high | Excellent | 1K-10K |
| Crowdsource (MTurk) | Medium | Variable | 10K-100K |
| AI-assisted (RLAIF) | Low | Good | 100K+ |
| Automatic (code/math) | Very low | Perfect | Unlimited |

### Reward Hacking

The most dangerous failure mode is **reward hacking** — the policy finds outputs that score highly on the reward model but are actually low quality:

```
Example of reward hacking:
- RM learns that longer responses are usually preferred
- Policy generates extremely verbose, repetitive responses
- RM gives high scores, but actual quality is poor
```

**Mitigations for reward hacking:**
- Regularize with KL penalty against the reference policy
- Use ensemble of reward models
- Periodically refresh the RM with on-policy data
- Apply length normalization to reward scores
- Monitor proxy reward vs true performance (golden gate problem)

### Reward Model Architecture

Typically, a reward model is a language model with the final unembedding layer replaced by a scalar head:

```mermaid
graph LR
    A["Input tokens"] --> B["Transformer"]
    B --> C["[CLS] embedding"]
    C --> D["Linear"]
    D --> E["Scalar reward"]
```

The model shares the base architecture with the policy, enabling it to understand the same distribution of text. Training usually starts from the SFT checkpoint for better initialization.
""",
                },
                {
                    "title": "Self-refinement",
                    "content": """\
## Self-refinement

Self-refinement techniques enable models to **improve themselves** by generating their own training data and iteratively bootstrapping better reasoning capabilities, reducing dependence on human-annotated data.

### STaR (Self-Taught Reasoner)

STaR (Zelikman et al., 2022) is the foundational self-refinement algorithm:

```mermaid
graph TD
    A["1. Generate rationales\nfor training problems"] --> B["2. Filter: keep only those\nleading to correct answers"]
    B --> C["3. For failed problems, provide\nanswer as hint and re-generate"]
    C --> D["4. Fine-tune model on\nsuccessful rationales"]
    D --> E["5. Repeat from step 1\nwith the improved model"]
    E --> A
```

```python
def star_iteration(model, problems, answers):
    training_data = []
    for problem, answer in zip(problems, answers):
        # Try to solve
        rationale = model.generate(f"Solve: {problem}")
        if extract_answer(rationale) == answer:
            training_data.append((problem, rationale))
        else:
            # Rationalization — hint with answer
            rationale = model.generate(
                f"Solve: {problem}. The answer is {answer}."
            )
            if extract_answer(rationale) == answer:
                training_data.append((problem, rationale))

    # Fine-tune on successful rationales
    model.finetune(training_data)
    return model
```

### Self-Play

Inspired by game-playing AI, self-play pits the model against itself:

- **Generator** produces candidate solutions
- **Critic** (same model, different prompt) evaluates them
- Both roles improve as training progresses

This creates a **virtuous cycle**: better generation gives the critic harder examples, and better critique gives the generator more useful feedback.

### Iterative Self-Improvement Loops

Modern self-improvement combines multiple signals:

| Iteration | Training Data Source | Filtering |
|-----------|---------------------|-----------|
| Round 1 | Human demonstrations | Manual review |
| Round 2 | Model solutions, verified | Automated correctness check |
| Round 3 | Model self-critique + revision | Reward model scoring |
| Round N | Fully self-generated curriculum | Multi-signal verification |

### Risks and Safeguards

Self-refinement can **amplify errors** if not carefully managed:

- **Model collapse** — training on own outputs can reduce diversity
- **Error propagation** — incorrect solutions that happen to reach right answers poison training
- **Distributional shift** — self-generated data drifts from real-world distribution

Safeguards include mixing self-generated data with original human data, using formal verifiers where possible, and monitoring quality metrics across iterations.
""",
                },
                {
                    "title": "Internalizing search",
                    "content": """\
## Internalizing Search

Internalizing search is the process of **distilling explicit search and planning procedures into model weights**, so that the model performs implicit search during a single (extended) forward pass. This is the core idea behind OpenAI's o1/o3 and DeepSeek-R1.

### The Key Idea

Instead of running an external search algorithm (MCTS, beam search) at inference time, train the model to **simulate that search internally** using "thinking tokens":

```
Traditional approach:
  Prompt → [External MCTS + Verifier] → Answer
  (Expensive, requires infrastructure)

Internalized search:
  Prompt → <think>...extended internal reasoning...</think> → Answer
  (Single model call, search is implicit)
```

### How o1/o3 Work (Conceptual)

While OpenAI hasn't published full details, the likely training pipeline is:

1. **Start with a strong base model** (GPT-4 class)
2. **Generate search traces** — Run MCTS/tree search on hard problems, record the exploration process (attempts, backtracking, verification)
3. **Train on linearized search** — Convert tree search into sequential token streams
4. **RL optimization** — Use reinforcement learning to reward solutions that arrive at correct answers through thinking

```
<think>
Let me try approach A...
Hmm, that gives 23, which seems wrong because...
Let me backtrack and try approach B...
This gives 42. Let me verify: 42 × 2 = 84. Yes, correct.
</think>
The answer is 42.
```

### Thinking Tokens

The `<think>...</think>` block is a designated space for **scratchpad computation**. During training, the model learns to use this space for:

- Exploring alternative approaches
- Self-verification and error checking
- Breaking problems into subproblems
- Backtracking from dead ends

### DeepSeek-R1's Approach

DeepSeek-R1 demonstrated that **pure RL** (without SFT on reasoning traces) can produce internalized search:

| Training Stage | Technique | Result |
|---------------|-----------|--------|
| Cold start | Small SFT dataset | Basic reasoning format |
| RL training | GRPO with rule-based rewards | Emergent search behaviors |
| Rejection sampling | Filter best RL outputs | High-quality training data |
| Final SFT | Mix reasoning + general data | Balanced capabilities |

### The Compute Tradeoff

Internalized search trades **training compute for inference efficiency**. The model requires significantly more training (RL is expensive), but each inference call is a single forward pass — no external search infrastructure needed. The model dynamically allocates more thinking tokens to harder problems, providing adaptive compute allocation similar to explicit search but with lower latency.
""",
                },
            ],
        },
        {
            "title": "Local Deployment",
            "content": """\
## Local Deployment

Local deployment enables running large language models on personal hardware — from laptops to multi-GPU workstations. This provides privacy, eliminates API costs, enables offline use, and allows full control over the model.

### Quantization Methods

Quantization reduces model precision from 16-bit floats to lower bit-widths, dramatically reducing memory requirements:

| Format | Bits | Method | Memory (7B model) | Quality Loss | Best For |
|--------|------|--------|-------------------|--------------|----------|
| FP16 | 16 | None (baseline) | ~14 GB | None | Reference |
| GPTQ | 4 | Post-training, GPU | ~4 GB | Minimal | GPU inference |
| AWQ | 4 | Activation-aware | ~4 GB | Very low | GPU inference |
| GGUF | 4-8 | llama.cpp format | ~4-8 GB | Low | CPU + GPU |
| GGML | 4-8 | Legacy llama.cpp | ~4-8 GB | Low | CPU (deprecated) |

**GGUF** is the most versatile format, supporting CPU, GPU, and hybrid (partial offload) inference. **GPTQ** and **AWQ** are GPU-optimized and generally faster on CUDA hardware.

### Inference Engines

**llama.cpp** — C++ engine supporting GGUF models on CPU/GPU:
```bash
./main -m model.gguf -p "Hello" -n 256 --n-gpu-layers 35
```

**Ollama** — User-friendly wrapper around llama.cpp:
```bash
ollama pull llama3:8b
ollama run llama3:8b "Explain quantum computing"
```

**vLLM** — High-throughput GPU server with PagedAttention:
```bash
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3-8B-Instruct --dtype float16
```

### Hardware Requirements

| Model Size | Min RAM | Recommended GPU | Quantization |
|-----------|---------|-----------------|-------------|
| 1-3B | 4 GB | None (CPU ok) | Q4_K_M |
| 7-8B | 8 GB | RTX 3060 12GB | Q4_K_M |
| 13B | 16 GB | RTX 3090 24GB | Q4_K_M |
| 34B | 32 GB | RTX 4090 24GB | Q4_K_M |
| 70B | 64 GB | 2× RTX 4090 or A100 | Q4_K_M |

### Choosing the Right Setup

- **Privacy-first**: Ollama + GGUF on local hardware
- **Maximum throughput**: vLLM on GPU server
- **Experimentation**: llama.cpp with various quantization levels
- **Production API**: vLLM with OpenAI-compatible endpoint

Local deployment is essential for building deep research systems that iterate rapidly without API rate limits or costs.
""",
            "children": [],
        },
    ],
}

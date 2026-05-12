PROJECT_6 = {
    "number": 6,
    "title": "Ship a Portfolio-Ready AI Project",
    "description": "Capstone: Ship a Portfolio-Ready AI Project",
    "topics": [
        {
            "title": "Choose idea",
            "content": """## Choosing Your Capstone Idea

Your capstone project is the centerpiece of your AI portfolio. Choosing the right idea means balancing ambition with feasibility so you can deliver a polished, working product.

### The Feasibility Matrix

Evaluate every idea on two axes: **impact** and **effort**.

| | Low Effort | High Effort |
|---|---|---|
| **High Impact** | ✅ Sweet spot — start here | ⚠️ Impressive but risky |
| **Low Impact** | 🔄 Too trivial for a capstone | ❌ Avoid entirely |

Aim for high-impact, low-to-moderate effort ideas. These let you ship something meaningful without drowning in complexity.

### Novelty vs Practicality

A common mistake is chasing novelty at the expense of practicality. Recruiters and collaborators care more about a **well-executed solution to a real problem** than a flashy idea that barely works. Apply AI where it genuinely adds value rather than forcing it into a project for the sake of using it.

### Strong Capstone Examples

- **Domain-specific chatbot**: A customer support bot trained on a company's documentation that answers questions accurately with source citations.
- **Code review agent**: An AI tool that analyzes pull requests, identifies potential bugs, and suggests improvements using LLM-powered reasoning.
- **Research assistant**: A RAG-based application that ingests academic papers and answers questions with referenced summaries.
- **Content moderation pipeline**: A system that classifies user-generated content using fine-tuned models and escalates edge cases for human review.

### Scoping Your MVP

Define the smallest version of your project that still demonstrates real value. Write down three things your MVP **must** do and three things it **won't** do yet. This boundary prevents scope creep, the number-one killer of capstone projects.

### Validating Your Idea

Before writing code, describe your project to two or three potential users in one sentence. If they immediately understand the value and ask follow-up questions, you have a strong concept. If you need a lengthy explanation, simplify the scope or reconsider the idea entirely.

### Checklist Before Starting

- [ ] Problem is clearly defined in one sentence
- [ ] Target user is identified
- [ ] MVP scope is written down with explicit boundaries
- [ ] Required data or APIs are accessible
- [ ] You can build a basic prototype within two weeks
""",
            "children": [],
        },
        {
            "title": "Build implementation",
            "content": """## Building Your Implementation

Building an AI project requires a disciplined development workflow that accounts for the unique challenges of non-deterministic systems.

### Iterative Prototyping Approach

Start with the simplest possible pipeline and improve incrementally:

1. **Prototype** — Hard-code inputs, use the simplest model, prove the concept works end to end.
2. **Harden** — Add error handling, swap in better models, connect real data sources.
3. **Polish** — Optimize performance, improve UX, add monitoring and logging.

Never jump to step three. Each stage should produce a working system you can demo.

### Testing AI Systems

AI systems blend deterministic and non-deterministic components. Test them differently:

| Component Type | Testing Strategy | Example |
|---|---|---|
| Deterministic | Unit tests, integration tests | Data parsing, API routing |
| Non-deterministic | Evaluation sets, assertion-based checks | LLM outputs, classification results |

Build an **evaluation set** of 20-50 input-output pairs early. Run it after every major change to catch regressions. Use assertions like "output must contain keyword X" or "response length must be under 500 tokens" to add structure to non-deterministic testing.

### Version Control for ML

Standard Git is not enough for ML projects. Track data and experiments alongside code:

- **DVC (Data Version Control)**: Version large datasets and model files without bloating your Git repo. Use `dvc push` and `dvc pull` to sync with remote storage.
- **MLflow**: Log experiment parameters, metrics, and artifacts. Compare runs side by side to understand what changed and why.
- **Prompt versioning**: Store prompts in version-controlled files, not inline strings. Name versions explicitly (e.g., `summarize_v3.txt`) so you can roll back reliably.

### CI/CD for ML Pipelines

Automate your workflow with continuous integration:

- Run your evaluation set on every pull request.
- Lint prompt templates for formatting errors.
- Validate data schemas before training or inference.
- Deploy with feature flags so you can roll back without redeploying.

### Documentation-Driven Development

Write your README **before** building. Describe what the project does, how to run it, and what the expected outputs look like. This forces clarity of thought and gives you a living document that evolves with your code. Include environment setup instructions using tools like `pip-compile`, `conda`, or Docker to ensure reproducibility.
""",
            "children": [],
        },
        {
            "title": "Iterate with feedback",
            "content": """## Iterating with Feedback

Shipping a great AI project is not a one-shot effort. Systematic feedback loops separate polished portfolio pieces from rough prototypes.

### User Testing Methodologies

Run structured user tests with real people, not just yourself:

- **Think-aloud testing**: Ask users to narrate their thoughts while using your app. You will discover confusion you never anticipated.
- **Task-based testing**: Give users a specific goal (e.g., "Find the answer to X using the chatbot") and observe where they struggle.
- **Wizard of Oz testing**: Before building expensive features, simulate them manually to validate demand.

Aim for **five testers minimum**. Research shows that five users uncover roughly 80% of usability issues.

### A/B Testing Prompts and Models

When you have two candidate approaches, test them head to head:

| Element | Variant A | Variant B | Metric |
|---|---|---|---|
| System prompt | Concise instructions | Detailed instructions | Answer accuracy |
| Model | GPT-4o-mini | Claude Haiku | Latency + quality |
| Retrieval | Top-3 chunks | Top-5 chunks | Relevance score |

Log every variant and outcome so you can make data-driven decisions rather than guessing.

### Collecting Structured Feedback

Do not just ask "What do you think?" Instead, use structured formats:

- **Likert scales**: "Rate the helpfulness of this response from 1 to 5."
- **Binary judgments**: "Was this answer correct? Yes or No."
- **Free-text with prompts**: "What would have made this response better?"

Store feedback in a simple database or spreadsheet so you can analyze trends over time.

### Quantitative vs Qualitative Metrics

Balance both types of measurement:

- **Quantitative**: accuracy, latency, token usage, user retention, task completion rate.
- **Qualitative**: user satisfaction comments, confusion points, feature requests.

Quantitative metrics tell you **what** is happening. Qualitative feedback tells you **why**.

### Knowing When to Stop

Iteration without a stopping condition leads to burnout. Define exit criteria upfront:

- [ ] Evaluation set accuracy exceeds your target threshold
- [ ] Three consecutive testers complete the core task without assistance
- [ ] No critical bugs remain in the backlog
- [ ] Edge cases are handled with graceful fallback messages

When these conditions are met, stop iterating and move to presentation. Perfection is the enemy of shipped.
""",
            "children": [],
        },
        {
            "title": "Demo presentation",
            "content": """## Demo and Presentation

A brilliant project that is poorly presented will not land the way it deserves. Learn to showcase your AI work effectively to both technical and non-technical audiences.

### Demo Structure

Follow this proven four-part structure:

1. **Problem** (30 seconds): State the pain point clearly. "Developers spend 2 hours per day reviewing code manually."
2. **Solution** (1 minute): Explain your approach at a high level. "I built a code review agent that uses RAG to surface relevant style guidelines."
3. **Live demo** (3-5 minutes): Show the working product. Use pre-selected inputs that highlight your system's strengths, but also show one realistic edge case.
4. **Impact and learnings** (1 minute): Share metrics, user feedback, and what you would do differently.

### Handling AI Failures During Demos

AI systems can produce unexpected outputs live. Prepare for this:

- **Pre-cache key examples**: Have a recorded backup for your critical demo path.
- **Narrate failures positively**: "This is a great example of why we built the fallback mechanism — let me show you how it recovers."
- **Set expectations upfront**: "This is a probabilistic system, so let me show you both a strong response and how we handle weaker ones."

### Portfolio Best Practices

Your project lives beyond the demo. Make it discoverable and impressive:

| Asset | Purpose | Tips |
|---|---|---|
| **GitHub README** | First impression for visitors | Include problem statement, architecture diagram, setup instructions, and a GIF or screenshot |
| **Live demo** | Proves it actually works | Deploy on a free tier (Streamlit Cloud, Hugging Face Spaces, Vercel) |
| **Video walkthrough** | Reaches people who will not clone your repo | Record a 3-5 minute Loom or YouTube video showing the full workflow |
| **Technical blog post** | Demonstrates depth of understanding | Write about one interesting challenge you solved and the trade-offs involved |

### Writing a Technical Blog Post

Structure your post around a single insight or challenge:

- **Hook**: Start with the problem, not the solution.
- **Context**: What did you try first and why did it fall short?
- **Solution**: Walk through your approach with code snippets.
- **Results**: Share concrete metrics or before-and-after comparisons.

### Presenting to Different Audiences

- **Technical audience**: Lead with architecture, trade-offs, and evaluation metrics. Show code when relevant.
- **Non-technical audience**: Lead with the user problem and the impact. Skip implementation details and focus on outcomes and the demo.

Tailor your language, but never oversimplify to the point of inaccuracy. Respect your audience by being clear and honest about what your system can and cannot do.
""",
            "children": [],
        },
    ],
}

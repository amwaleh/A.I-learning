# Project 4: Build "Deep Research" — Hands-on Tutorial

## What You'll Build

A **Deep Research** system that:
1. Takes a complex research question
2. Breaks it into sub-questions
3. Searches the web iteratively for each sub-question
4. Uses chain-of-thought reasoning to synthesize findings
5. Implements Tree of Thoughts for complex reasoning
6. Produces a comprehensive research report with citations
7. Optionally runs entirely locally with Ollama

```
┌─────────────────────────────────────────────────────────────┐
│                    DEEP RESEARCH SYSTEM                      │
│                                                             │
│  "What are the long-term      ┌──────────────────┐         │
│   effects of microplastics    │ Question Planner │         │
│   on human health?"           └────────┬─────────┘         │
│                                        │                   │
│                    ┌───────────────────┼──────────────┐    │
│                    ▼                   ▼              ▼    │
│              Sub-Q 1            Sub-Q 2          Sub-Q 3   │
│              "Sources of        "Known health    "Current  │
│               exposure"          effects"        research" │
│                    │                   │              │    │
│                    ▼                   ▼              ▼    │
│              ┌──────────┐       ┌──────────┐   ┌────────┐ │
│              │Web Search│       │Web Search│   │Web Srch│ │
│              └────┬─────┘       └────┬─────┘   └───┬────┘ │
│                   │                  │             │       │
│                   └──────────────────┼─────────────┘       │
│                                      ▼                     │
│                            ┌─────────────────┐             │
│                            │  Reasoning &    │             │
│                            │  Synthesis      │             │
│                            │  (Tree of       │             │
│                            │   Thoughts)     │             │
│                            └────────┬────────┘             │
│                                     ▼                      │
│                            ┌─────────────────┐             │
│                            │ Research Report │             │
│                            │ with Citations  │             │
│                            └─────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Programming language |
| pip | Latest | Package manager |
| Ollama | Latest | Local model hosting (optional) |

### API Keys Needed

| Service | Free Tier? | Purpose |
|---------|-----------|---------|
| OpenAI API | Pay-per-use | Reasoning model (GPT-4.1 / o4-mini) |
| Tavily API | 1000 free searches/month | Web search |

**Get your API keys:**
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://tavily.com (sign up → dashboard → API key)

---

## Environment Setup

### Step 1: Create Project Directory

```bash
mkdir deep-research
cd deep-research
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install openai tavily-python ollama python-dotenv
```

### Step 4: Create Environment File

Create a file named `.env` in your project directory:

```env
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here
```

### Step 5: Verify Setup

Create `test_setup.py`:

```python
"""Verify all dependencies are installed and API keys work."""
import os
from dotenv import load_dotenv

load_dotenv()

# Check packages
print("Checking packages...")
import openai
print(f"  ✓ openai {openai.__version__}")

from tavily import TavilyClient
print(f"  ✓ tavily-python installed")

print("\nChecking API keys...")
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if openai_key and openai_key != "sk-your-openai-key-here":
    print(f"  ✓ OPENAI_API_KEY set ({openai_key[:8]}...)")
else:
    print("  ✗ OPENAI_API_KEY not set — add it to .env")

if tavily_key and tavily_key != "tvly-your-tavily-key-here":
    print(f"  ✓ TAVILY_API_KEY set ({tavily_key[:8]}...)")
else:
    print("  ✗ TAVILY_API_KEY not set — add it to .env")

print("\n✅ Setup complete! You're ready to build.")
```

Run it:

```bash
python test_setup.py
```

**Expected output:**
```
Checking packages...
  ✓ openai 1.x.x
  ✓ tavily-python installed

Checking API keys...
  ✓ OPENAI_API_KEY set (sk-proj-...)
  ✓ TAVILY_API_KEY set (tvly-...)

✅ Setup complete! You're ready to build.
```

---

## Milestone 1: Basic Web Search

We'll start simple — just search the web and display results.

Create `step1_search.py`:

```python
"""Step 1: Basic web search with Tavily."""
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Initialize Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return results.
    
    Returns a list of dicts with keys: title, url, content
    """
    response = tavily.search(
        query=query,
        max_results=max_results,
        include_answer=True
    )
    
    results = []
    for result in response.get("results", []):
        results.append({
            "title": result["title"],
            "url": result["url"],
            "content": result["content"]
        })
    
    return results


def display_results(query: str, results: list[dict]):
    """Pretty-print search results."""
    print(f"\n{'='*60}")
    print(f"Search: {query}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['title']}")
        print(f"    URL: {result['url']}")
        print(f"    {result['content'][:150]}...")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    query = "What are the health effects of microplastics on humans 2024"
    results = search_web(query)
    display_results(query, results)
```

**Run it:**
```bash
python step1_search.py
```

**Expected output:**
```
============================================================
Search: What are the health effects of microplastics on humans 2024
============================================================

[1] Microplastics found in human blood for first time
    URL: https://...
    Research shows microplastics have been detected in human bloodstreams...

[2] Health impacts of microplastic exposure
    URL: https://...
    Studies indicate potential links between microplastic exposure and...

...
============================================================
```

### Common Pitfalls — Step 1

| Problem | Solution |
|---------|----------|
| `tavily.errors.InvalidAPIKeyError` | Check your TAVILY_API_KEY in `.env` |
| `ModuleNotFoundError: No module named 'tavily'` | Run `pip install tavily-python` |
| Empty results | Try a different query; some topics have limited coverage |

---

## Milestone 2: Question Decomposition with LLM

Now we'll use an LLM to break a complex question into sub-questions.

Create `step2_planner.py`:

```python
"""Step 2: Break complex questions into sub-questions using an LLM."""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def decompose_question(question: str, num_subquestions: int = 4) -> list[str]:
    """
    Break a complex research question into focused sub-questions.
    
    Uses chain-of-thought prompting to ensure good decomposition.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a research planning assistant. 
Your job is to break complex questions into focused, searchable sub-questions.

Rules:
- Each sub-question should be independently searchable on the web
- Cover different aspects of the main question
- Order them logically (background first, then specifics, then implications)
- Make them specific enough to get good search results

Respond with a JSON array of strings. Nothing else."""
            },
            {
                "role": "user",
                "content": f"""Break this research question into {num_subquestions} focused sub-questions:

"{question}"

Think step by step about what aspects need to be researched, then provide the sub-questions as a JSON array."""
            }
        ],
        temperature=0.3
    )
    
    # Parse the JSON response
    content = response.choices[0].message.content.strip()
    # Handle markdown code blocks if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    
    sub_questions = json.loads(content)
    return sub_questions


def display_plan(question: str, sub_questions: list[str]):
    """Display the research plan."""
    print(f"\n{'='*60}")
    print(f"RESEARCH PLAN")
    print(f"{'='*60}")
    print(f"\nMain Question: {question}")
    print(f"\nSub-questions to research:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    question = "What are the long-term effects of microplastics on human health?"
    
    print("Decomposing question...")
    sub_questions = decompose_question(question)
    display_plan(question, sub_questions)
```

**Run it:**
```bash
python step2_planner.py
```

**Expected output:**
```
Decomposing question...

============================================================
RESEARCH PLAN
============================================================

Main Question: What are the long-term effects of microplastics on human health?

Sub-questions to research:
  1. What are microplastics and how do humans get exposed to them?
  2. What do current studies show about microplastics in human organs and blood?
  3. What specific health conditions have been linked to microplastic exposure?
  4. What are researchers predicting about long-term cumulative effects?

============================================================
```

---

## Milestone 3: Iterative Research Loop

Now we combine search + LLM to research each sub-question iteratively.

Create `step3_researcher.py`:

```python
"""Step 3: Iterative research - search and synthesize for each sub-question."""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return results."""
    response = tavily.search(query=query, max_results=max_results)
    results = []
    for result in response.get("results", []):
        results.append({
            "title": result["title"],
            "url": result["url"],
            "content": result["content"]
        })
    return results


def decompose_question(question: str) -> list[str]:
    """Break a complex question into sub-questions."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Break the question into 4 focused, searchable sub-questions. Respond with a JSON array of strings only."
            },
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)


def research_subquestion(sub_question: str) -> dict:
    """
    Research a single sub-question:
    1. Search the web
    2. Synthesize findings with chain-of-thought
    3. Identify if follow-up search is needed
    4. Return findings with sources
    """
    print(f"\n  Researching: {sub_question}")
    
    # First search
    results = search_web(sub_question)
    print(f"  Found {len(results)} sources")
    
    # Format search results for the LLM
    search_context = ""
    sources = []
    for i, r in enumerate(results, 1):
        search_context += f"\n[Source {i}]: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n"
        sources.append({"title": r["title"], "url": r["url"]})
    
    # Synthesize with chain-of-thought reasoning
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a research analyst. Synthesize the search results 
to answer the question. Think step by step:
1. Identify the key facts from each source
2. Note any contradictions or gaps
3. Synthesize a clear, comprehensive answer
4. Note what's still uncertain or needs more research

If the information is insufficient, suggest a follow-up search query.

Format your response as JSON:
{
    "thinking": "your step-by-step reasoning",
    "findings": "synthesized answer (2-3 paragraphs)",
    "confidence": "high/medium/low",
    "follow_up_query": "null or a follow-up search query if needed"
}"""
            },
            {
                "role": "user",
                "content": f"Question: {sub_question}\n\nSearch Results:{search_context}"
            }
        ],
        temperature=0.2
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    
    synthesis = json.loads(content)
    
    # Follow-up search if needed
    if synthesis.get("follow_up_query") and synthesis["confidence"] != "high":
        print(f"  Doing follow-up search: {synthesis['follow_up_query']}")
        follow_up_results = search_web(synthesis["follow_up_query"], max_results=3)
        
        # Add follow-up results to sources
        for r in follow_up_results:
            sources.append({"title": r["title"], "url": r["url"]})
        
        # Re-synthesize with additional information
        additional_context = ""
        for i, r in enumerate(follow_up_results, 1):
            additional_context += f"\n[Additional Source {i}]: {r['title']}\nContent: {r['content']}\n"
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Update your findings with this additional information. Return only the updated findings as plain text (2-3 paragraphs)."
                },
                {
                    "role": "user",
                    "content": f"Original findings: {synthesis['findings']}\n\nAdditional information:{additional_context}"
                }
            ],
            temperature=0.2
        )
        synthesis["findings"] = response.choices[0].message.content.strip()
    
    return {
        "question": sub_question,
        "findings": synthesis["findings"],
        "confidence": synthesis.get("confidence", "medium"),
        "sources": sources
    }


def run_research(question: str) -> list[dict]:
    """Run the full iterative research process."""
    print(f"\n{'='*60}")
    print(f"DEEP RESEARCH: {question}")
    print(f"{'='*60}")
    
    # Decompose
    print("\nStep 1: Decomposing question...")
    sub_questions = decompose_question(question)
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    
    # Research each sub-question
    print("\nStep 2: Researching each sub-question...")
    all_findings = []
    for sq in sub_questions:
        finding = research_subquestion(sq)
        all_findings.append(finding)
        print(f"  ✓ Done (confidence: {finding['confidence']})")
    
    return all_findings


if __name__ == "__main__":
    question = "What are the long-term effects of microplastics on human health?"
    findings = run_research(question)
    
    # Display findings
    print(f"\n\n{'='*60}")
    print("RESEARCH FINDINGS SUMMARY")
    print(f"{'='*60}")
    
    for i, finding in enumerate(findings, 1):
        print(f"\n--- Finding {i}: {finding['question']} ---")
        print(f"Confidence: {finding['confidence']}")
        print(f"Sources: {len(finding['sources'])}")
        print(f"\n{finding['findings'][:300]}...")
```

**Run it:**
```bash
python step3_researcher.py
```

**Expected output:**
```
============================================================
DEEP RESEARCH: What are the long-term effects of microplastics on human health?
============================================================

Step 1: Decomposing question...
  1. What are microplastics and how do humans get exposed to them?
  2. What do current studies show about microplastics in human organs?
  3. What health conditions have been linked to microplastic exposure?
  4. What are the predicted long-term cumulative health effects?

Step 2: Researching each sub-question...

  Researching: What are microplastics and how do humans get exposed to them?
  Found 5 sources
  ✓ Done (confidence: high)

  Researching: What do current studies show about microplastics in human organs?
  Found 5 sources
  Doing follow-up search: microplastics detected human brain lung tissue 2024
  ✓ Done (confidence: medium)
  ...
```

---

## Milestone 4: Tree of Thoughts Reasoning

For complex synthesis, we use Tree of Thoughts to explore multiple reasoning paths.

Create `step4_tree_of_thoughts.py`:

```python
"""Step 4: Tree of Thoughts for complex reasoning and synthesis."""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ThoughtNode:
    """A single node in the thought tree."""
    
    def __init__(self, content: str, score: float = 0.0, parent=None):
        self.content = content
        self.score = score
        self.parent = parent
        self.children = []
    
    def get_path(self) -> str:
        """Get the full reasoning path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node.content)
            node = node.parent
        return "\n→ ".join(reversed(path))


def generate_thoughts(problem: str, context: str, num_thoughts: int = 3) -> list[str]:
    """Generate multiple next-step thoughts."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": f"""You are reasoning about a complex problem step by step.

Problem: {problem}

Reasoning so far:
{context if context else "(Starting fresh - generate initial approaches)"}

Generate exactly {num_thoughts} DIFFERENT possible next reasoning steps.
Each should take a distinct approach or angle.

Respond as a JSON array of {num_thoughts} strings."""
        }],
        temperature=0.8
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [content]


def evaluate_thought(problem: str, thought_path: str) -> float:
    """Score a reasoning path from 0.0 to 1.0."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": f"""Rate this reasoning path for solving the problem.

Problem: {problem}

Reasoning path:
{thought_path}

Score from 0.0 to 1.0 where:
- 0.0 = completely wrong or irrelevant
- 0.5 = somewhat useful but incomplete
- 1.0 = excellent, making strong progress

Respond with ONLY a decimal number (e.g., 0.7). Nothing else."""
        }],
        temperature=0.1
    )
    
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.5


def tree_of_thoughts(
    problem: str,
    breadth: int = 3,
    depth: int = 3,
    beam_width: int = 2
) -> str:
    """
    Explore multiple reasoning paths using Tree of Thoughts.
    
    Args:
        problem: The problem to solve
        breadth: Number of thoughts to generate at each step
        depth: How many reasoning steps deep to go
        beam_width: How many best paths to keep at each level
    
    Returns:
        The best reasoning path found
    """
    print(f"\n  Tree of Thoughts (breadth={breadth}, depth={depth})")
    
    # Level 0: Generate initial thoughts
    initial_thoughts = generate_thoughts(problem, "", breadth)
    
    # Create root nodes
    current_nodes = []
    for thought in initial_thoughts:
        node = ThoughtNode(content=thought)
        node.score = evaluate_thought(problem, thought)
        current_nodes.append(node)
        print(f"    L0: [{node.score:.2f}] {thought[:60]}...")
    
    # Expand tree level by level
    for level in range(1, depth):
        # Keep only the best nodes (beam search)
        current_nodes.sort(key=lambda n: n.score, reverse=True)
        best_nodes = current_nodes[:beam_width]
        
        print(f"    Keeping top {beam_width} paths at level {level}")
        
        # Expand each of the best nodes
        next_nodes = []
        for node in best_nodes:
            path = node.get_path()
            new_thoughts = generate_thoughts(problem, path, breadth)
            
            for thought in new_thoughts:
                child = ThoughtNode(content=thought, parent=node)
                child.score = evaluate_thought(problem, child.get_path())
                node.children.append(child)
                next_nodes.append(child)
                print(f"    L{level}: [{child.score:.2f}] {thought[:50]}...")
        
        current_nodes = next_nodes
    
    # Select the best final path
    current_nodes.sort(key=lambda n: n.score, reverse=True)
    best_node = current_nodes[0]
    best_path = best_node.get_path()
    
    print(f"\n  Best path score: {best_node.score:.2f}")
    
    return best_path


def synthesize_with_tot(problem: str, research_findings: list[dict]) -> str:
    """
    Use Tree of Thoughts to synthesize research findings into a coherent analysis.
    """
    # Format findings for context
    findings_text = ""
    for i, f in enumerate(research_findings, 1):
        findings_text += f"\n[Finding {i}]: {f['question']}\n{f['findings']}\n"
    
    synthesis_problem = f"""Based on these research findings, synthesize a comprehensive 
answer to: {problem}

Research findings:
{findings_text}

Create a well-structured analysis that:
1. Connects findings across sub-topics
2. Identifies key themes and patterns
3. Notes areas of uncertainty
4. Draws evidence-based conclusions"""
    
    # Use ToT to find the best synthesis approach
    best_reasoning = tree_of_thoughts(
        problem=synthesis_problem,
        breadth=3,
        depth=2,
        beam_width=2
    )
    
    # Generate final synthesis using the best reasoning path
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a research analyst producing a final synthesis. Write clearly and cite sources by number."
            },
            {
                "role": "user",
                "content": f"""Problem: {problem}

Research findings:
{findings_text}

Best reasoning approach:
{best_reasoning}

Now write the final synthesized analysis (3-5 paragraphs). 
Reference findings by number (e.g., [Finding 1])."""
            }
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    # Test with mock findings
    mock_findings = [
        {
            "question": "What are microplastics?",
            "findings": "Microplastics are tiny plastic particles less than 5mm. They come from degradation of larger plastics, synthetic clothing fibers, and industrial processes. Humans ingest them through water, food, and air."
        },
        {
            "question": "What health effects have been observed?",
            "findings": "Studies show microplastics in human blood, lungs, and placenta. Potential effects include inflammation, oxidative stress, and endocrine disruption. A 2024 study linked them to cardiovascular events."
        },
        {
            "question": "What are the long-term predictions?",
            "findings": "Researchers predict cumulative bioaccumulation effects. Long-term exposure may increase cancer risk and affect reproductive health. However, definitive long-term studies in humans are still lacking."
        }
    ]
    
    print("Running Tree of Thoughts synthesis...")
    result = synthesize_with_tot(
        "What are the long-term effects of microplastics on human health?",
        mock_findings
    )
    
    print(f"\n{'='*60}")
    print("SYNTHESIZED ANALYSIS")
    print(f"{'='*60}")
    print(result)
```

**Run it:**
```bash
python step4_tree_of_thoughts.py
```

**Expected output:**
```
Running Tree of Thoughts synthesis...

  Tree of Thoughts (breadth=3, depth=2)
    L0: [0.75] Start by categorizing effects into short-term and long-te...
    L0: [0.80] First identify the biological mechanisms, then connect to...
    L0: [0.65] Compare microplastic effects to other environmental pollu...
    Keeping top 2 paths at level 1
    L1: [0.85] The evidence suggests a pathway: ingestion → bloodstream...
    L1: [0.70] ...
    ...

  Best path score: 0.85

============================================================
SYNTHESIZED ANALYSIS
============================================================

The research reveals a concerning picture of microplastic exposure...
[Full synthesized analysis appears here]
```

---

## Milestone 5: Report Generation with Citations

Create `step5_report.py`:

```python
"""Step 5: Generate a comprehensive research report with proper citations."""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_report(
    question: str,
    findings: list[dict],
    synthesis: str
) -> str:
    """
    Generate a formatted research report with citations.
    
    Args:
        question: The original research question
        findings: List of research findings with sources
        synthesis: The synthesized analysis from ToT
    
    Returns:
        Formatted markdown report
    """
    # Collect all unique sources
    all_sources = []
    source_map = {}  # url -> citation number
    
    for finding in findings:
        for source in finding.get("sources", []):
            url = source["url"]
            if url not in source_map:
                source_map[url] = len(all_sources) + 1
                all_sources.append(source)
    
    # Generate the executive summary
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Write a concise 2-3 sentence executive summary of the research findings. Be direct and factual."
            },
            {
                "role": "user",
                "content": f"Research question: {question}\n\nFindings synthesis:\n{synthesis}"
            }
        ],
        temperature=0.2
    )
    executive_summary = response.choices[0].message.content.strip()
    
    # Build the report
    report = f"""# Deep Research Report

## Question
{question}

## Executive Summary
{executive_summary}

---

## Detailed Findings

"""
    
    # Add each finding section
    for i, finding in enumerate(findings, 1):
        report += f"### {i}. {finding['question']}\n\n"
        report += f"{finding['findings']}\n\n"
        
        if finding.get("sources"):
            report += "**Sources:**\n"
            for source in finding["sources"]:
                num = source_map.get(source["url"], "?")
                report += f"- [{num}] {source['title']}\n"
            report += "\n"
        
        report += f"*Confidence: {finding.get('confidence', 'medium')}*\n\n"
        report += "---\n\n"
    
    # Add synthesis section
    report += f"""## Synthesis & Analysis

{synthesis}

---

## References

"""
    
    for i, source in enumerate(all_sources, 1):
        report += f"[{i}] {source['title']}\n    {source['url']}\n\n"
    
    # Add metadata
    report += f"""---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Model: GPT-4.1-mini with Tree of Thoughts reasoning*
*Sources consulted: {len(all_sources)}*
"""
    
    return report


def save_report(report: str, filename: str = None):
    """Save report to a markdown file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Report saved to: {filename}")
    return filename


if __name__ == "__main__":
    # Demo with mock data
    mock_findings = [
        {
            "question": "What are microplastics and how are humans exposed?",
            "findings": "Microplastics are plastic particles smaller than 5mm. Primary sources of human exposure include drinking water (both tap and bottled), food packaging, seafood consumption, and airborne particles from synthetic textiles. Studies estimate humans ingest approximately 5 grams of plastic per week.",
            "confidence": "high",
            "sources": [
                {"title": "WHO Report on Microplastics in Drinking Water", "url": "https://www.who.int/publications/microplastics-water"},
                {"title": "Nature: Microplastic Ingestion Estimates", "url": "https://www.nature.com/articles/microplastic-ingestion"}
            ]
        },
        {
            "question": "What health effects have been linked to microplastics?",
            "findings": "Research has linked microplastic exposure to inflammatory responses, oxidative stress, and potential endocrine disruption. A landmark 2024 study in the New England Journal of Medicine found that patients with microplastics in arterial plaque had a 4.5x higher risk of cardiovascular events.",
            "confidence": "medium",
            "sources": [
                {"title": "NEJM: Microplastics and Cardiovascular Risk", "url": "https://www.nejm.org/doi/microplastics-cardiovascular"},
                {"title": "Environmental Health Perspectives: Endocrine Effects", "url": "https://ehp.niehs.nih.gov/microplastics-endocrine"}
            ]
        },
        {
            "question": "What are predicted long-term effects?",
            "findings": "Long-term predictions include bioaccumulation in organs, potential cancer risk from chronic inflammation, and reproductive health impacts. Animal studies show transgenerational effects. However, long-term human epidemiological data is still being collected.",
            "confidence": "low",
            "sources": [
                {"title": "Science: Transgenerational Effects of Microplastics", "url": "https://www.science.org/transgenerational-microplastics"},
            ]
        }
    ]
    
    mock_synthesis = """The evidence paints a concerning picture of microplastic exposure and human health. 
Based on the research findings, three key themes emerge:

First, exposure is ubiquitous and unavoidable — humans ingest significant quantities weekly through 
normal daily activities [Finding 1]. Second, the biological mechanisms of harm are becoming clearer, 
with inflammation and cardiovascular effects now supported by clinical evidence [Finding 2]. Third, 
the long-term trajectory suggests cumulative harm, though definitive proof requires ongoing study [Finding 3].

The strongest evidence currently links microplastics to cardiovascular disease, while long-term cancer 
and reproductive effects remain plausible but less certain. The gap between animal study findings and 
human epidemiology represents the key area needing further research."""
    
    report = generate_report(
        "What are the long-term effects of microplastics on human health?",
        mock_findings,
        mock_synthesis
    )
    
    # Display and save
    print(report)
    save_report(report, "example_report.md")
```

**Run it:**
```bash
python step5_report.py
```

**Expected output:** A complete markdown report printed to console and saved to `example_report.md`.

---

## Milestone 6: Complete Deep Research System

Now we combine everything into a single, cohesive system.

Create `deep_research.py`:

```python
"""
Deep Research System — Complete Implementation

Combines:
- Question decomposition (LLM planning)
- Web search (Tavily)
- Iterative research with follow-ups
- Tree of Thoughts reasoning
- Report generation with citations

Usage:
    python deep_research.py "Your complex research question here"
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL = "gpt-4.1-mini"            # LLM model to use
SEARCH_RESULTS_PER_QUERY = 5      # Web results per search
NUM_SUB_QUESTIONS = 4             # How many sub-questions to generate
TOT_BREADTH = 3                   # Tree of Thoughts: branches per level
TOT_DEPTH = 2                     # Tree of Thoughts: levels of depth
TOT_BEAM_WIDTH = 2                # Tree of Thoughts: paths to keep

# ─── Clients ──────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ─── Web Search ───────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = SEARCH_RESULTS_PER_QUERY) -> list[dict]:
    """Search the web using Tavily."""
    try:
        response = tavily.search(query=query, max_results=max_results)
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response.get("results", [])
        ]
    except Exception as e:
        print(f"    ⚠ Search error: {e}")
        return []


# ─── Question Decomposition ──────────────────────────────────────────────────

def decompose_question(question: str) -> list[str]:
    """Break a complex question into focused sub-questions."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""Break the research question into exactly {NUM_SUB_QUESTIONS} focused sub-questions.
Each should be:
- Independently searchable on the web
- Covering a different aspect of the main question
- Ordered logically (background → specifics → implications)
Respond with a JSON array of strings only."""
            },
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)


# ─── Research a Sub-Question ──────────────────────────────────────────────────

def research_subquestion(sub_question: str) -> dict:
    """Research a sub-question with web search and synthesis."""
    print(f"\n  📎 {sub_question}")
    
    # Search
    results = search_web(sub_question)
    if not results:
        return {
            "question": sub_question,
            "findings": "No search results found for this query.",
            "confidence": "low",
            "sources": []
        }
    
    print(f"    Found {len(results)} sources")
    
    # Build context
    search_context = "\n".join(
        f"[{i}] {r['title']} ({r['url']})\n{r['content']}"
        for i, r in enumerate(results, 1)
    )
    sources = [{"title": r["title"], "url": r["url"]} for r in results]
    
    # Synthesize with CoT
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """Synthesize the search results to answer the question.
Think step by step:
1. Identify key facts from each source
2. Note contradictions or gaps
3. Write a clear 2-3 paragraph synthesis

Respond as JSON: {"findings": "...", "confidence": "high/medium/low", "follow_up_query": null or "..."}"""
            },
            {
                "role": "user",
                "content": f"Question: {sub_question}\n\nSearch Results:\n{search_context}"
            }
        ],
        temperature=0.2
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    
    try:
        synthesis = json.loads(content)
    except json.JSONDecodeError:
        synthesis = {"findings": content, "confidence": "medium", "follow_up_query": None}
    
    # Follow-up search if needed
    follow_up = synthesis.get("follow_up_query")
    if follow_up and synthesis.get("confidence") != "high":
        print(f"    🔄 Follow-up: {follow_up[:50]}...")
        extra_results = search_web(follow_up, max_results=3)
        for r in extra_results:
            sources.append({"title": r["title"], "url": r["url"]})
        
        if extra_results:
            extra_context = "\n".join(r["content"] for r in extra_results)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Update the findings with new information. Return plain text (2-3 paragraphs)."},
                    {"role": "user", "content": f"Current findings: {synthesis['findings']}\n\nNew info:\n{extra_context}"}
                ],
                temperature=0.2
            )
            synthesis["findings"] = response.choices[0].message.content.strip()
    
    print(f"    ✓ Confidence: {synthesis.get('confidence', 'medium')}")
    
    return {
        "question": sub_question,
        "findings": synthesis["findings"],
        "confidence": synthesis.get("confidence", "medium"),
        "sources": sources
    }


# ─── Tree of Thoughts Synthesis ──────────────────────────────────────────────

def tot_generate_thoughts(problem: str, context: str) -> list[str]:
    """Generate multiple reasoning approaches."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"""Problem: {problem}

Progress so far: {context if context else "(Starting fresh)"}

Generate {TOT_BREADTH} DIFFERENT next reasoning steps as a JSON array of strings."""
        }],
        temperature=0.8
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [content]


def tot_evaluate(problem: str, path: str) -> float:
    """Score a reasoning path 0.0-1.0."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Problem: {problem}\n\nReasoning: {path}\n\nScore 0.0-1.0 (just the number):"
        }],
        temperature=0.1
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.5


def tree_of_thoughts_synthesis(question: str, findings: list[dict]) -> str:
    """Use ToT to find the best synthesis approach, then generate the synthesis."""
    findings_text = "\n".join(
        f"[Finding {i}] {f['question']}: {f['findings']}"
        for i, f in enumerate(findings, 1)
    )
    
    problem = f"Synthesize research findings into a comprehensive answer to: {question}\n\nFindings:\n{findings_text}"
    
    print("\n  🌳 Running Tree of Thoughts...")
    
    # Level 0: Initial approaches
    thoughts = tot_generate_thoughts(problem, "")
    scored = [(t, tot_evaluate(problem, t)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    for t, s in scored:
        print(f"    [{s:.2f}] {t[:60]}...")
    
    # Level 1+: Expand best paths
    for level in range(1, TOT_DEPTH):
        best = scored[:TOT_BEAM_WIDTH]
        next_scored = []
        
        for path, _ in best:
            new_thoughts = tot_generate_thoughts(problem, path)
            for thought in new_thoughts:
                full_path = f"{path}\n→ {thought}"
                score = tot_evaluate(problem, full_path)
                next_scored.append((full_path, score))
        
        scored = sorted(next_scored, key=lambda x: x[1], reverse=True)
        print(f"    Level {level} best: [{scored[0][1]:.2f}]")
    
    best_path = scored[0][0]
    
    # Generate final synthesis
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Write a comprehensive research synthesis. Reference findings by number. Be analytical and evidence-based."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nFindings:\n{findings_text}\n\nBest reasoning approach:\n{best_path}\n\nWrite 3-5 paragraphs of synthesis:"
            }
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()


# ─── Report Generation ────────────────────────────────────────────────────────

def generate_report(question: str, findings: list[dict], synthesis: str) -> str:
    """Generate a complete markdown research report."""
    # Collect sources
    all_sources = []
    source_map = {}
    for finding in findings:
        for source in finding.get("sources", []):
            if source["url"] not in source_map:
                source_map[source["url"]] = len(all_sources) + 1
                all_sources.append(source)
    
    # Executive summary
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Write a 2-3 sentence executive summary. Be direct and factual."},
            {"role": "user", "content": f"Question: {question}\nSynthesis:\n{synthesis}"}
        ],
        temperature=0.2
    )
    summary = response.choices[0].message.content.strip()
    
    # Build report
    report = f"""# Deep Research Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Model:** {MODEL} with Tree of Thoughts reasoning  
**Sources consulted:** {len(all_sources)}

---

## Research Question

> {question}

## Executive Summary

{summary}

---

## Detailed Findings

"""
    
    for i, finding in enumerate(findings, 1):
        report += f"### {i}. {finding['question']}\n\n"
        report += f"{finding['findings']}\n\n"
        if finding.get("sources"):
            report += "**Sources:** "
            nums = [str(source_map.get(s["url"], "?")) for s in finding["sources"]]
            report += ", ".join(f"[{n}]" for n in nums) + "\n\n"
        report += f"*Confidence: {finding.get('confidence', 'medium')}*\n\n---\n\n"
    
    report += f"""## Synthesis & Analysis

{synthesis}

---

## References

"""
    for i, source in enumerate(all_sources, 1):
        report += f"{i}. [{source['title']}]({source['url']})\n"
    
    return report


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def deep_research(question: str) -> str:
    """
    Run the complete deep research pipeline.
    
    1. Decompose question
    2. Research each sub-question (with web search)
    3. Synthesize with Tree of Thoughts
    4. Generate report
    """
    print(f"\n{'='*60}")
    print(f"🔬 DEEP RESEARCH")
    print(f"{'='*60}")
    print(f"\n  Question: {question}\n")
    
    # Step 1: Decompose
    print("📋 Step 1: Planning research...")
    sub_questions = decompose_question(question)
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    
    # Step 2: Research
    print("\n🔍 Step 2: Researching sub-questions...")
    findings = []
    for sq in sub_questions:
        finding = research_subquestion(sq)
        findings.append(finding)
    
    # Step 3: Synthesize with ToT
    print("\n🧠 Step 3: Synthesizing with Tree of Thoughts...")
    synthesis = tree_of_thoughts_synthesis(question, findings)
    
    # Step 4: Generate report
    print("\n📝 Step 4: Generating report...")
    report = generate_report(question, findings, synthesis)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"✅ Research complete! Report saved to: {filename}")
    print(f"{'='*60}")
    
    return report


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the long-term effects of microplastics on human health?"
    
    report = deep_research(question)
    print("\n" + report)
```

**Run it:**
```bash
python deep_research.py "What are the implications of quantum computing for current encryption methods?"
```

**Expected output:**
```
============================================================
🔬 DEEP RESEARCH
============================================================

  Question: What are the implications of quantum computing for current encryption methods?

📋 Step 1: Planning research...
  1. How do current encryption methods (RSA, AES, ECC) work?
  2. What quantum computing capabilities threaten current encryption?
  3. What is post-quantum cryptography and which algorithms are recommended?
  4. What is the timeline for quantum computers breaking current encryption?

🔍 Step 2: Researching sub-questions...
  📎 How do current encryption methods work?
    Found 5 sources
    ✓ Confidence: high
  ...

🧠 Step 3: Synthesizing with Tree of Thoughts...
  🌳 Running Tree of Thoughts...
    [0.80] Organize by threat level...
    [0.75] Compare quantum vs classical...
    ...
    Level 1 best: [0.85]

📝 Step 4: Generating report...

============================================================
✅ Research complete! Report saved to: research_report_20260512_113800.md
============================================================
```

---

## Milestone 7: Local Deployment with Ollama

Create `deep_research_local.py` — a version that runs entirely locally:

```python
"""
Deep Research System — Local Version (Ollama)

Runs entirely on your machine using Ollama for the LLM.
Still uses Tavily for web search (requires internet + API key).

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull deepseek-r1:8b
    3. Start Ollama: ollama serve

Usage:
    python deep_research_local.py "Your research question"
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

# Use Ollama's OpenAI-compatible endpoint
LOCAL_MODEL = "deepseek-r1:8b"     # Change based on what you've pulled
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# For simpler tasks (planning, evaluation), use a faster model
FAST_MODEL = "llama3.2:3b"         # Smaller, faster model for simple tasks

# Reduce ToT parameters for local (slower) models
TOT_BREADTH = 2
TOT_DEPTH = 2
TOT_BEAM_WIDTH = 1

# ─── Clients ──────────────────────────────────────────────────────────────────

# Local LLM client (Ollama)
local_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # Ollama doesn't need a real key
)

# Web search still uses Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ─── Helper: Check Ollama is Running ─────────────────────────────────────────

def check_ollama():
    """Verify Ollama is running and model is available."""
    try:
        response = local_client.models.list()
        models = [m.id for m in response.data]
        
        if LOCAL_MODEL not in models:
            print(f"⚠ Model '{LOCAL_MODEL}' not found. Available models:")
            for m in models:
                print(f"    - {m}")
            print(f"\nPull it with: ollama pull {LOCAL_MODEL}")
            sys.exit(1)
        
        print(f"✓ Ollama running with {LOCAL_MODEL}")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Install: https://ollama.ai")
        print(f"  2. Pull model: ollama pull {LOCAL_MODEL}")
        print("  3. Start server: ollama serve")
        sys.exit(1)


# ─── LLM Calls (Local) ───────────────────────────────────────────────────────

def llm_call(prompt: str, system: str = "", model: str = None, temperature: float = 0.3) -> str:
    """Make a call to the local LLM."""
    model = model or LOCAL_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = local_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ⚠ LLM error: {e}")
        return ""


def llm_json(prompt: str, system: str = "") -> dict | list:
    """Make a call expecting JSON response."""
    content = llm_call(prompt, system, temperature=0.2)
    
    # Try to extract JSON from response
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    
    # Find JSON in the response
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = content.find(start_char)
        end = content.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                continue
    
    # Fallback: return as single-item list
    return [content]


# ─── Web Search ───────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily."""
    try:
        response = tavily.search(query=query, max_results=max_results)
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response.get("results", [])
        ]
    except Exception as e:
        print(f"    ⚠ Search error: {e}")
        return []


# ─── Research Pipeline (Local) ────────────────────────────────────────────────

def decompose_question(question: str) -> list[str]:
    """Break question into sub-questions using local model."""
    result = llm_json(
        prompt=f"Break this into 3-4 searchable sub-questions: {question}",
        system="Respond with a JSON array of strings. Each string is a focused sub-question."
    )
    if isinstance(result, list):
        return result[:4]
    return [question]


def research_subquestion(sub_question: str) -> dict:
    """Research a sub-question using web search + local LLM."""
    print(f"\n  📎 {sub_question}")
    
    results = search_web(sub_question)
    if not results:
        return {"question": sub_question, "findings": "No results found.", "confidence": "low", "sources": []}
    
    print(f"    Found {len(results)} sources")
    sources = [{"title": r["title"], "url": r["url"]} for r in results]
    
    # Synthesize
    context = "\n".join(f"Source: {r['title']}\n{r['content']}" for r in results)
    
    findings = llm_call(
        prompt=f"Question: {sub_question}\n\nSources:\n{context}\n\nSynthesize 2-3 paragraphs answering the question based on these sources.",
        system="You are a research analyst. Synthesize information clearly and accurately."
    )
    
    # Remove <think> blocks if present (DeepSeek-R1 includes them)
    if "<think>" in findings:
        think_end = findings.find("</think>")
        if think_end != -1:
            findings = findings[think_end + 8:].strip()
    
    print(f"    ✓ Synthesized")
    
    return {
        "question": sub_question,
        "findings": findings,
        "confidence": "medium",
        "sources": sources
    }


def synthesize_findings(question: str, findings: list[dict]) -> str:
    """Synthesize all findings into a cohesive analysis."""
    findings_text = "\n\n".join(
        f"### {f['question']}\n{f['findings']}"
        for f in findings
    )
    
    synthesis = llm_call(
        prompt=f"""Research question: {question}

Individual findings:
{findings_text}

Write a comprehensive synthesis (3-5 paragraphs) that:
1. Connects themes across findings
2. Identifies key conclusions
3. Notes uncertainties
Reference findings where relevant.""",
        system="You are a senior research analyst writing a synthesis. Be thorough and analytical."
    )
    
    # Remove <think> blocks
    if "<think>" in synthesis:
        think_end = synthesis.find("</think>")
        if think_end != -1:
            synthesis = synthesis[think_end + 8:].strip()
    
    return synthesis


def generate_report(question: str, findings: list[dict], synthesis: str) -> str:
    """Generate the final markdown report."""
    all_sources = []
    for f in findings:
        for s in f.get("sources", []):
            if s not in all_sources:
                all_sources.append(s)
    
    report = f"""# Deep Research Report (Local)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Model:** {LOCAL_MODEL} (Ollama - running locally)  
**Sources consulted:** {len(all_sources)}

---

## Research Question

> {question}

---

## Detailed Findings

"""
    for i, f in enumerate(findings, 1):
        report += f"### {i}. {f['question']}\n\n{f['findings']}\n\n"
        if f.get("sources"):
            report += "**Sources:**\n"
            for s in f["sources"]:
                report += f"- [{s['title']}]({s['url']})\n"
            report += "\n"
        report += "---\n\n"
    
    report += f"""## Synthesis

{synthesis}

---

## References

"""
    for i, s in enumerate(all_sources, 1):
        report += f"{i}. [{s['title']}]({s['url']})\n"
    
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def deep_research_local(question: str) -> str:
    """Run the full local deep research pipeline."""
    print(f"\n{'='*60}")
    print(f"🔬 DEEP RESEARCH (Local - {LOCAL_MODEL})")
    print(f"{'='*60}")
    print(f"\n  Question: {question}\n")
    
    # Check Ollama
    check_ollama()
    
    # Step 1: Decompose
    print("\n📋 Step 1: Planning...")
    sub_questions = decompose_question(question)
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    
    # Step 2: Research
    print("\n🔍 Step 2: Researching...")
    findings = [research_subquestion(sq) for sq in sub_questions]
    
    # Step 3: Synthesize
    print("\n🧠 Step 3: Synthesizing...")
    synthesis = synthesize_findings(question, findings)
    
    # Step 4: Report
    print("\n📝 Step 4: Generating report...")
    report = generate_report(question, findings, synthesis)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"local_report_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✅ Done! Report: {filename}")
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the current developments in nuclear fusion energy?"
    
    report = deep_research_local(question)
    print("\n" + report)
```

**Run it:**
```bash
# First, make sure Ollama is running with a model
ollama pull deepseek-r1:8b
ollama serve  # In a separate terminal

# Then run:
python deep_research_local.py "What are the current developments in nuclear fusion energy?"
```

---

## Troubleshooting Guide

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `openai.AuthenticationError` | Invalid API key | Check `.env` file, ensure no extra spaces |
| `openai.RateLimitError` | Too many requests | Add `time.sleep(1)` between API calls |
| `json.JSONDecodeError` | LLM didn't return valid JSON | The code handles this with fallbacks; try lowering temperature |
| `Connection refused (Ollama)` | Ollama not running | Run `ollama serve` in another terminal |
| `Model not found (Ollama)` | Model not downloaded | Run `ollama pull deepseek-r1:8b` |
| `Out of memory (Ollama)` | Model too large for your RAM | Use a smaller model: `ollama pull deepseek-r1:1.5b` |
| Slow responses (Ollama) | CPU-only inference | Normal for CPU; use GPU if available |
| Empty search results | Tavily rate limit or bad query | Check API key; try simpler query |

### Tips for Better Results

1. **Better questions get better research:**
   - ✗ "Tell me about AI"
   - ✓ "What are the latest breakthroughs in protein folding prediction using AI in 2024?"

2. **Adjust parameters for your hardware:**
   ```python
   # For slower machines, reduce complexity:
   TOT_BREADTH = 2      # fewer branches
   TOT_DEPTH = 1        # shallower tree
   NUM_SUB_QUESTIONS = 3 # fewer searches
   ```

3. **Local model selection:**
   - 1.5B: Very fast, lower quality (good for testing)
   - 8B: Good balance of speed and quality
   - 14B+: Better quality, needs more RAM/VRAM

4. **Save money on API calls during development:**
   - Use `gpt-4.1-mini` instead of `gpt-4.1` (much cheaper)
   - Test with mock data first (like in step 4 and 5)
   - Cache search results during development

---

## Project Structure (Final)

```
deep-research/
├── .env                      # API keys (DO NOT commit this!)
├── test_setup.py             # Verify environment
├── step1_search.py           # Milestone 1: Basic web search
├── step2_planner.py          # Milestone 2: Question decomposition
├── step3_researcher.py       # Milestone 3: Iterative research
├── step4_tree_of_thoughts.py # Milestone 4: ToT reasoning
├── step5_report.py           # Milestone 5: Report generation
├── deep_research.py          # Milestone 6: Complete system (API)
├── deep_research_local.py    # Milestone 7: Local version (Ollama)
└── research_report_*.md      # Generated reports
```

---

## Next Steps & Extensions

Once you have the basic system working, try these enhancements:

1. **Add a UI** — Use Streamlit or Gradio for a web interface
2. **Parallel research** — Use `asyncio` to research sub-questions concurrently
3. **Source quality scoring** — Rate sources by authority/recency
4. **Multi-hop research** — Follow up on findings that reference other topics
5. **PDF/paper analysis** — Add ArXiv paper search and summarization
6. **Conversation mode** — Let users ask follow-up questions about the report
7. **Compare models** — Run same question through different models and compare

---

## Summary

You've built a Deep Research system that demonstrates:

| Concept | Where It's Used |
|---------|----------------|
| Chain-of-Thought | Every LLM synthesis call uses step-by-step reasoning |
| Tree of Thoughts | Milestone 4 & 6: Finding best synthesis approach |
| Iterative search | Milestone 3: Follow-up queries when confidence is low |
| Question decomposition | Breaking complex questions into searchable parts |
| Local deployment | Milestone 7: Running everything through Ollama |
| Report generation | Milestone 5 & 6: Structured output with citations |

Congratulations! You now understand how reasoning models work and have built a practical system that uses them. 🎉

# Project 1: Build an LLM Playground — Hands-on Tutorial

Build a local LLM playground where you can load models, adjust generation parameters, compare outputs, and visualize tokenization.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Step 1: Hello, Transformers!](#3-step-1-hello-transformers)
4. [Step 2: Visualize Tokenization](#4-step-2-visualize-tokenization)
5. [Step 3: Explore Generation Parameters](#5-step-3-explore-generation-parameters)
6. [Step 4: Build the Interactive Playground](#6-step-4-build-the-interactive-playground)
7. [Step 5: Compare Multiple Models](#7-step-5-compare-multiple-models)
8. [Step 6: Full Playground Application](#8-step-6-full-playground-application)
9. [Common Pitfalls & Troubleshooting](#9-common-pitfalls--troubleshooting)

---

## 1. Prerequisites

**Knowledge:**
- Basic Python (variables, functions, loops, f-strings)
- Comfort using the command line / terminal

**Hardware:**
- At minimum: 8 GB RAM (for small models like `distilgpt2`)
- Recommended: 16+ GB RAM and a GPU for larger models
- Disk: ~2-5 GB free for model downloads

**Software:**
- Python 3.9 or newer
- pip (Python package installer)
- A text editor or IDE (VS Code recommended)

---

## 2. Environment Setup

### Step 2.1: Create a Project Folder

```bash
mkdir llm-playground
cd llm-playground
```

### Step 2.2: Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 2.3: Install Dependencies

```bash
pip install transformers torch rich
```

| Package        | Purpose                                           |
|----------------|---------------------------------------------------|
| `transformers` | Hugging Face library to load and run LLMs         |
| `torch`        | PyTorch — the deep learning framework             |
| `rich`         | Beautiful terminal output (colored text, tables)  |

### Step 2.4: Verify Installation

Create a file called `check_setup.py`:

```python
"""Verify that all dependencies are installed correctly."""

import transformers
import torch
from rich import print as rprint

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
rprint("[green]✓ All dependencies installed successfully![/green]")
```

Run it:

```bash
python check_setup.py
```

**Expected output:**
```
Transformers version: 4.x.x
PyTorch version: 2.x.x
CUDA available: True (or False if no GPU)
✓ All dependencies installed successfully!
```

---

## 3. Step 1: Hello, Transformers!

Let's generate our first text with an LLM.

### Create `step1_hello.py`:

```python
"""Step 1: Generate text with a pre-trained language model."""

from transformers import pipeline

# Create a text generation pipeline with a small model
# distilgpt2 is ~82M parameters — small enough to run on any machine
generator = pipeline("text-generation", model="distilgpt2")

# Generate text from a prompt
prompt = "Once upon a time in a land far away"
result = generator(prompt, max_length=50, num_return_sequences=1)

print("=" * 60)
print("PROMPT:", prompt)
print("=" * 60)
print("GENERATED TEXT:")
print(result[0]["generated_text"])
print("=" * 60)
```

Run it:

```bash
python step1_hello.py
```

**Expected output** (your text will differ — that's normal!):
```
============================================================
PROMPT: Once upon a time in a land far away
============================================================
GENERATED TEXT:
Once upon a time in a land far away, there lived a king who
ruled over all the creatures of the forest. He was known for
his wisdom and...
============================================================
```

> **Note:** The first run downloads the model (~350 MB). Subsequent runs use the cached version.

### What Just Happened?

```
pipeline("text-generation", model="distilgpt2")
    │
    ├── Downloaded the model weights
    ├── Downloaded the tokenizer
    └── Created a ready-to-use text generator

generator(prompt, max_length=50)
    │
    ├── Tokenized the prompt into token IDs
    ├── Fed tokens through the neural network
    ├── Generated new tokens one at a time
    └── Decoded token IDs back to text
```

---

## 4. Step 2: Visualize Tokenization

Understanding tokenization is key to understanding LLMs. Let's build a visual tokenizer.

### Create `step2_tokenizer.py`:

```python
"""Step 2: Visualize how text is tokenized."""

from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

# Color palette for highlighting tokens
COLORS = [
    "red", "green", "blue", "yellow", "magenta",
    "cyan", "bright_red", "bright_green", "bright_blue", "bright_yellow",
]


def visualize_tokens(text, model_name="distilgpt2"):
    """Show how a tokenizer breaks text into tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    # Header
    console.print(f"\n[bold]Model:[/bold] {model_name}")
    console.print(f"[bold]Input:[/bold] \"{text}\"")
    console.print(f"[bold]Number of tokens:[/bold] {len(tokens)}\n")

    # Colored token visualization
    colored = Text()
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        # Replace the special space character for display
        display_token = token.replace("Ġ", "·")
        colored.append(f"[{display_token}]", style=f"bold {color}")
        colored.append(" ")
    console.print("Tokens:", colored)

    # Token table
    table = Table(title="Token Details")
    table.add_column("Index", style="dim")
    table.add_column("Token", style="bold")
    table.add_column("Token ID")
    table.add_column("Characters")

    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        display_token = token.replace("Ġ", "·(space)")
        table.add_row(str(i), display_token, str(token_id), str(len(token)))

    console.print(table)


# Run examples
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TOKENIZATION VISUALIZER")
    print("=" * 60)

    examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",
        "supercalifragilisticexpialidocious",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]

    for text in examples:
        visualize_tokens(text)
        console.print("─" * 60)
```

Run it:

```bash
python step2_tokenizer.py
```

**Expected output:**
```
============================================================
TOKENIZATION VISUALIZER
============================================================

Model: distilgpt2
Input: "Hello, world!"
Number of tokens: 4

Tokens: [Hello] [,] [·world] [!]

┌───────────────────── Token Details ─────────────────────┐
│ Index │ Token          │ Token ID │ Characters           │
├───────┼────────────────┼──────────┼──────────────────────┤
│ 0     │ Hello          │ 15496    │ 5                    │
│ 1     │ ,              │ 11       │ 1                    │
│ 2     │ ·(space)world  │ 995      │ 6                    │
│ 3     │ !              │ 0        │ 1                    │
└───────┴────────────────┴──────────┴──────────────────────┘
────────────────────────────────────────────────────────────
```

### Key Observations

- Common words like "Hello" are single tokens
- Rare/long words get split: "supercalifragilistic..." becomes multiple tokens
- Spaces are attached to the beginning of tokens (shown as `Ġ` or `·`)
- Code gets tokenized differently than English text

---

## 5. Step 3: Explore Generation Parameters

Now let's see how different parameters affect text generation.

### Create `step3_parameters.py`:

```python
"""Step 3: Explore how generation parameters affect output."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Load model and tokenizer
MODEL_NAME = "distilgpt2"
console.print(f"[dim]Loading model: {MODEL_NAME}...[/dim]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set pad token (distilgpt2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token


def generate(prompt, **kwargs):
    """Generate text with given parameters."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    
    # Decode only the generated part (not the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def compare_parameter(prompt, param_name, values, fixed_params=None):
    """Generate text with different values of a parameter and compare."""
    if fixed_params is None:
        fixed_params = {}

    console.print(f"\n[bold cyan]Comparing: {param_name}[/bold cyan]")
    console.print(f"[dim]Prompt: \"{prompt}\"[/dim]\n")

    table = Table(title=f"Effect of {param_name}")
    table.add_column(param_name, style="bold")
    table.add_column("Generated Text")

    for value in values:
        params = {**fixed_params, param_name: value}
        text = generate(prompt, **params)
        table.add_row(str(value), text.strip()[:100])

    console.print(table)


if __name__ == "__main__":
    prompt = "The future of artificial intelligence is"

    # --- Temperature ---
    console.print(Panel("[bold]EXPERIMENT 1: Temperature[/bold]"))
    compare_parameter(
        prompt,
        "temperature",
        [0.1, 0.5, 1.0, 1.5, 2.0],
        fixed_params={"do_sample": True, "top_k": 50},
    )

    # --- Top-k ---
    console.print(Panel("[bold]EXPERIMENT 2: Top-k Sampling[/bold]"))
    compare_parameter(
        prompt,
        "top_k",
        [1, 5, 10, 50, 100],
        fixed_params={"do_sample": True, "temperature": 0.8},
    )

    # --- Top-p ---
    console.print(Panel("[bold]EXPERIMENT 3: Top-p (Nucleus) Sampling[/bold]"))
    compare_parameter(
        prompt,
        "top_p",
        [0.1, 0.3, 0.5, 0.9, 0.99],
        fixed_params={"do_sample": True, "temperature": 0.8},
    )

    # --- Greedy vs Beam Search ---
    console.print(Panel("[bold]EXPERIMENT 4: Greedy vs Beam Search[/bold]"))

    greedy = generate(prompt, do_sample=False)
    beam = generate(prompt, do_sample=False, num_beams=5)
    sample = generate(prompt, do_sample=True, temperature=0.8, top_k=50)

    table = Table(title="Search Strategies")
    table.add_column("Strategy", style="bold")
    table.add_column("Generated Text")
    table.add_row("Greedy (deterministic)", greedy.strip()[:100])
    table.add_row("Beam Search (n=5)", beam.strip()[:100])
    table.add_row("Sampling (temp=0.8)", sample.strip()[:100])
    console.print(table)

    console.print("\n[bold green]Key Takeaways:[/bold green]")
    console.print("• Low temperature → repetitive, safe text")
    console.print("• High temperature → creative but potentially nonsensical")
    console.print("• Low top-k → focused output (similar to greedy)")
    console.print("• High top-k → more variety")
    console.print("• Top-p adapts: confident predictions use fewer tokens")
```

Run it:

```bash
python step3_parameters.py
```

**Expected output** (text varies due to sampling):
```
Loading model: distilgpt2...

╭─ EXPERIMENT 1: Temperature ─╮
│                              │
╰──────────────────────────────╯

Comparing: temperature
Prompt: "The future of artificial intelligence is"

┌────────── Effect of temperature ──────────┐
│ temperature │ Generated Text              │
├─────────────┼─────────────────────────────┤
│ 0.1         │ going to be a lot more...   │
│ 0.5         │ going to be very different  │
│ 1.0         │ not just about the tech...  │
│ 1.5         │ already being written by... │
│ 2.0         │ redefin complimentary hub   │  ← nonsensical
└─────────────┴─────────────────────────────┘
```

---

## 6. Step 4: Build the Interactive Playground

Let's make it interactive so you can type prompts and adjust parameters in real time.

### Create `step4_interactive.py`:

```python
"""Step 4: Interactive LLM Playground in the terminal."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.table import Table
from rich.text import Text

console = Console()

# Color palette for token visualization
COLORS = [
    "red", "green", "blue", "yellow", "magenta",
    "cyan", "bright_red", "bright_green", "bright_blue", "bright_yellow",
]


class LLMPlayground:
    """An interactive LLM playground for experimenting with text generation."""

    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.params = {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
            "max_new_tokens": 50,
            "do_sample": True,
            "num_beams": 1,
        }
        self.load_model(model_name)

    def load_model(self, model_name):
        """Load a model and tokenizer."""
        console.print(f"\n[dim]Loading model: {model_name}...[/dim]")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        console.print(f"[green]✓ Model loaded: {model_name}[/green]")

        # Show model info
        param_count = sum(p.numel() for p in self.model.parameters())
        console.print(f"  Parameters: {param_count:,}")
        console.print(f"  Vocabulary size: {self.tokenizer.vocab_size:,}")

    def generate(self, prompt):
        """Generate text with current parameters."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.params["max_new_tokens"],
                temperature=self.params["temperature"],
                top_k=self.params["top_k"],
                top_p=self.params["top_p"],
                do_sample=self.params["do_sample"],
                num_beams=self.params["num_beams"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def visualize_tokens(self, text):
        """Show colorized token breakdown."""
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)

        colored = Text()
        for i, token in enumerate(tokens):
            color = COLORS[i % len(COLORS)]
            display = token.replace("Ġ", "·")
            colored.append(f"[{display}]", style=f"bold {color}")
            colored.append(" ")

        console.print(f"  Tokens ({len(tokens)}):", colored)

    def show_params(self):
        """Display current parameter settings."""
        table = Table(title="Current Parameters")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        table.add_column("Description")

        descriptions = {
            "temperature": "Controls randomness (0.1=focused, 2.0=random)",
            "top_k": "Number of top tokens to consider",
            "top_p": "Cumulative probability threshold",
            "max_new_tokens": "Maximum tokens to generate",
            "do_sample": "Enable random sampling (False=greedy)",
            "num_beams": "Beam search width (1=no beam search)",
        }

        for key, value in self.params.items():
            table.add_row(key, str(value), descriptions.get(key, ""))

        console.print(table)

    def adjust_params(self):
        """Interactive parameter adjustment."""
        self.show_params()
        console.print("\n[bold]Which parameter to change?[/bold]")
        console.print("Options: temperature, top_k, top_p, max_new_tokens, do_sample, num_beams")
        console.print("(press Enter to keep current values)")

        param = Prompt.ask("Parameter", default="")
        if not param or param not in self.params:
            return

        if param == "do_sample":
            val = Prompt.ask("Value (true/false)", default=str(self.params[param]))
            self.params[param] = val.lower() == "true"
        elif param in ("top_k", "max_new_tokens", "num_beams"):
            val = IntPrompt.ask("Value", default=self.params[param])
            self.params[param] = val
        else:
            val = FloatPrompt.ask("Value", default=self.params[param])
            self.params[param] = val

        console.print(f"[green]✓ {param} = {self.params[param]}[/green]")

    def run(self):
        """Main interactive loop."""
        console.print(Panel(
            "[bold cyan]LLM PLAYGROUND[/bold cyan]\n"
            "Experiment with language model text generation\n\n"
            "Commands:\n"
            "  [bold]/params[/bold]   - View/adjust generation parameters\n"
            "  [bold]/tokens[/bold]   - Tokenize your next input\n"
            "  [bold]/model[/bold]    - Switch model\n"
            "  [bold]/help[/bold]     - Show this help message\n"
            "  [bold]/quit[/bold]     - Exit the playground\n\n"
            "Or type any text to generate a continuation!",
            title="Welcome",
        ))

        tokenize_mode = False

        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]>>>[/bold yellow]")
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input == "/quit":
                console.print("[dim]Goodbye![/dim]")
                break
            elif user_input == "/params":
                self.adjust_params()
                continue
            elif user_input == "/tokens":
                tokenize_mode = True
                console.print("[dim]Next input will be tokenized. Type your text:[/dim]")
                continue
            elif user_input == "/model":
                name = Prompt.ask(
                    "Model name",
                    default="distilgpt2",
                )
                self.load_model(name)
                continue
            elif user_input == "/help":
                console.print(
                    "/params - adjust parameters\n"
                    "/tokens - tokenize text\n"
                    "/model  - switch model\n"
                    "/quit   - exit"
                )
                continue

            # Tokenize mode
            if tokenize_mode:
                self.visualize_tokens(user_input)
                tokenize_mode = False
                continue

            # Generate text
            console.print(f"[dim]Generating with: temp={self.params['temperature']}, "
                         f"top_k={self.params['top_k']}, top_p={self.params['top_p']}[/dim]")

            generated = self.generate(user_input)
            console.print(Panel(
                f"[bold]{user_input}[/bold][green]{generated}[/green]",
                title="Generated Text",
            ))

            # Show token count
            full_text = user_input + generated
            tokens = self.tokenizer.tokenize(full_text)
            console.print(f"[dim]Total tokens: {len(tokens)}[/dim]")


if __name__ == "__main__":
    playground = LLMPlayground()
    playground.run()
```

Run it:

```bash
python step4_interactive.py
```

**Expected output:**
```
╭─────────────────── Welcome ───────────────────╮
│ LLM PLAYGROUND                                │
│ Experiment with language model text generation │
│                                               │
│ Commands:                                     │
│   /params   - View/adjust generation params   │
│   /tokens   - Tokenize your next input        │
│   /model    - Switch model                    │
│   /quit     - Exit the playground             │
│                                               │
│ Or type any text to generate a continuation!  │
╰───────────────────────────────────────────────╯

>>> The meaning of life is
Generating with: temp=1.0, top_k=50, top_p=0.9
╭─────────── Generated Text ───────────────╮
│ The meaning of life is to be found in    │
│ the pursuit of happiness and knowledge   │
╰──────────────────────────────────────────╯
Total tokens: 18
```

---

## 7. Step 5: Compare Multiple Models

Let's build a tool that generates text from multiple models side by side.

### Create `step5_compare.py`:

```python
"""Step 5: Compare outputs from multiple models side by side."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table

console = Console()

# Models to compare (all small enough to run locally)
MODELS = [
    "distilgpt2",       # ~82M params - very fast
    "gpt2",            # ~124M params - slightly better
    # Uncomment these if you have more RAM/GPU:
    # "gpt2-medium",   # ~355M params
    # "gpt2-large",    # ~774M params
]


class ModelComparator:
    """Load multiple models and compare their outputs."""

    def __init__(self, model_names):
        self.models = {}
        self.tokenizers = {}

        for name in model_names:
            console.print(f"[dim]Loading {name}...[/dim]")
            self.tokenizers[name] = AutoTokenizer.from_pretrained(name)
            self.models[name] = AutoModelForCausalLM.from_pretrained(name)
            self.tokenizers[name].pad_token = self.tokenizers[name].eos_token
            param_count = sum(p.numel() for p in self.models[name].parameters())
            console.print(f"  [green]✓[/green] {name} ({param_count / 1e6:.0f}M params)")

    def generate(self, model_name, prompt, **kwargs):
        """Generate text from a specific model."""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        inputs = tokenizer(prompt, return_tensors="pt")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=kwargs.get("max_new_tokens", 50),
                temperature=kwargs.get("temperature", 0.8),
                top_k=kwargs.get("top_k", 50),
                top_p=kwargs.get("top_p", 0.9),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)

        return {
            "text": text,
            "time": elapsed,
            "tokens": tokens_generated,
            "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
        }

    def compare(self, prompt, **kwargs):
        """Generate from all models and display comparison."""
        console.print(f"\n[bold]Prompt:[/bold] \"{prompt}\"")
        console.print(f"[dim]Params: temp={kwargs.get('temperature', 0.8)}, "
                     f"top_k={kwargs.get('top_k', 50)}, "
                     f"top_p={kwargs.get('top_p', 0.9)}[/dim]\n")

        panels = []
        for name in self.models:
            result = self.generate(name, prompt, **kwargs)
            panel_text = (
                f"{result['text'].strip()}\n\n"
                f"[dim]─────────────────────[/dim]\n"
                f"[dim]Tokens: {result['tokens']} | "
                f"Time: {result['time']:.2f}s | "
                f"Speed: {result['tokens_per_sec']:.1f} tok/s[/dim]"
            )
            panels.append(Panel(panel_text, title=f"[bold]{name}[/bold]", expand=True))

        console.print(Columns(panels))

    def compare_temperatures(self, prompt, temperatures=None):
        """Compare same model at different temperatures."""
        if temperatures is None:
            temperatures = [0.3, 0.7, 1.0, 1.5]

        model_name = list(self.models.keys())[0]
        console.print(f"\n[bold]Temperature comparison ({model_name}):[/bold]")
        console.print(f"Prompt: \"{prompt}\"\n")

        table = Table()
        table.add_column("Temp", style="bold")
        table.add_column("Generated Text")

        for temp in temperatures:
            result = self.generate(model_name, prompt, temperature=temp, top_k=50)
            table.add_row(str(temp), result["text"].strip()[:80])

        console.print(table)


if __name__ == "__main__":
    console.print(Panel("[bold cyan]MODEL COMPARATOR[/bold cyan]", expand=False))

    comparator = ModelComparator(MODELS)

    prompts = [
        "In the year 2050, humans will",
        "The best way to learn programming is",
        "Once upon a time, a robot discovered",
    ]

    for prompt in prompts:
        console.print("\n" + "═" * 60)
        comparator.compare(prompt)

    # Temperature comparison
    console.print("\n" + "═" * 60)
    comparator.compare_temperatures(
        "Science has proven that",
        temperatures=[0.2, 0.5, 1.0, 1.8],
    )
```

Run it:

```bash
python step5_compare.py
```

**Expected output:**
```
╭─ MODEL COMPARATOR ─╮
╰─────────────────────╯
Loading distilgpt2...
  ✓ distilgpt2 (82M params)
Loading gpt2...
  ✓ gpt2 (124M params)

════════════════════════════════════════════════════════════

Prompt: "In the year 2050, humans will"
Params: temp=0.8, top_k=50, top_p=0.9

┌──── distilgpt2 ────┐  ┌────── gpt2 ──────┐
│ be able to live on  │  │ have developed    │
│ Mars and travel...  │  │ technology to...  │
│                     │  │                   │
│ Tokens: 50          │  │ Tokens: 50        │
│ Time: 1.2s          │  │ Time: 2.1s        │
└─────────────────────┘  └───────────────────┘
```

---

## 8. Step 6: Full Playground Application

Finally, let's combine everything into a polished playground application.

### Create `playground.py`:

```python
"""
LLM Playground - Full Application
==================================
A complete interactive playground for experimenting with language models.

Features:
- Load and switch between models
- Adjust all generation parameters
- Compare outputs with different settings
- Visualize tokenization
- Track generation history
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.markdown import Markdown
from rich.columns import Columns

console = Console()

COLORS = [
    "red", "green", "blue", "yellow", "magenta",
    "cyan", "bright_red", "bright_green", "bright_blue", "bright_yellow",
]

HELP_TEXT = """
## Commands

| Command | Description |
|---------|-------------|
| `/generate` or just type text | Generate text from prompt |
| `/params` | View and adjust parameters |
| `/set <param> <value>` | Quick set parameter (e.g., `/set temperature 0.5`) |
| `/model <name>` | Load a different model |
| `/tokens <text>` | Visualize tokenization |
| `/compare <text>` | Generate 3 variations with different temperatures |
| `/history` | Show generation history |
| `/clear` | Clear history |
| `/help` | Show this help |
| `/quit` | Exit |

## Available Models (small, runs on CPU)

- `distilgpt2` — 82M params, fastest
- `gpt2` — 124M params, better quality
- `gpt2-medium` — 355M params, needs more RAM
- `gpt2-large` — 774M params, needs 4GB+ RAM

## Tips

- Start with low temperature (0.3-0.7) for factual text
- Use high temperature (1.0-1.5) for creative writing
- Set `top_k=1` for completely deterministic output
- Use `/compare` to see how temperature affects output
"""


class FullPlayground:
    """Complete LLM playground with all features."""

    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.history = []
        self.params = {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "max_new_tokens": 60,
            "do_sample": True,
            "num_beams": 1,
            "repetition_penalty": 1.1,
        }
        self._load_model(model_name)

    def _load_model(self, model_name):
        """Load model and tokenizer."""
        console.print(f"[dim]Loading {model_name}...[/dim]")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_name = model_name
            param_count = sum(p.numel() for p in self.model.parameters())
            console.print(f"[green]✓ Loaded {model_name} ({param_count / 1e6:.1f}M parameters)[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load {model_name}: {e}[/red]")
            if self.model is None:
                console.print("[red]No model loaded. Please try another model.[/red]")

    def _generate(self, prompt, **override_params):
        """Generate text and return result dict."""
        params = {**self.params, **override_params}
        inputs = self.tokenizer(prompt, return_tensors="pt")

        gen_kwargs = {
            "max_new_tokens": params["max_new_tokens"],
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": params["repetition_penalty"],
        }

        if params["do_sample"]:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": params["temperature"],
                "top_k": params["top_k"],
                "top_p": params["top_p"],
            })
        else:
            gen_kwargs["do_sample"] = False
            if params["num_beams"] > 1:
                gen_kwargs["num_beams"] = params["num_beams"]

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], **gen_kwargs)
        elapsed = time.time() - start

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        result = {
            "prompt": prompt,
            "text": text,
            "time": elapsed,
            "tokens": len(generated_ids),
            "model": self.model_name,
            "params": dict(params),
        }

        self.history.append(result)
        return result

    def cmd_generate(self, prompt):
        """Generate and display text."""
        if not prompt:
            prompt = Prompt.ask("[yellow]Enter prompt[/yellow]")
        if not prompt:
            return

        result = self._generate(prompt)
        console.print(Panel(
            f"[bold white]{prompt}[/bold white][green]{result['text']}[/green]",
            title=f"[bold]{self.model_name}[/bold]",
            subtitle=(
                f"[dim]{result['tokens']} tokens | "
                f"{result['time']:.2f}s | "
                f"{result['tokens'] / result['time']:.1f} tok/s[/dim]"
            ),
        ))

    def cmd_params(self):
        """Show and optionally modify parameters."""
        table = Table(title=f"Parameters ({self.model_name})")
        table.add_column("#", style="dim")
        table.add_column("Parameter", style="bold")
        table.add_column("Value", style="cyan")
        table.add_column("Description")

        descriptions = {
            "temperature": "Randomness (0.1=focused → 2.0=creative)",
            "top_k": "Top-k tokens to sample from",
            "top_p": "Nucleus sampling threshold",
            "max_new_tokens": "Max tokens to generate",
            "do_sample": "Random sampling (False=greedy/beam)",
            "num_beams": "Beam search width",
            "repetition_penalty": "Penalize repeated tokens (1.0=off)",
        }

        for i, (key, value) in enumerate(self.params.items(), 1):
            table.add_row(str(i), key, str(value), descriptions.get(key, ""))

        console.print(table)
        console.print("[dim]Use '/set <param> <value>' to change a parameter[/dim]")

    def cmd_set(self, args):
        """Set a parameter value."""
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            console.print("[red]Usage: /set <parameter> <value>[/red]")
            return

        param, value = parts
        if param not in self.params:
            console.print(f"[red]Unknown parameter: {param}[/red]")
            console.print(f"[dim]Available: {', '.join(self.params.keys())}[/dim]")
            return

        try:
            if param == "do_sample":
                self.params[param] = value.lower() in ("true", "1", "yes")
            elif param in ("top_k", "max_new_tokens", "num_beams"):
                self.params[param] = int(value)
            else:
                self.params[param] = float(value)
            console.print(f"[green]✓ {param} = {self.params[param]}[/green]")
        except ValueError:
            console.print(f"[red]Invalid value: {value}[/red]")

    def cmd_tokens(self, text):
        """Visualize tokenization."""
        if not text:
            text = Prompt.ask("[yellow]Enter text to tokenize[/yellow]")
        if not text:
            return

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)

        # Colored display
        colored = Text()
        for i, token in enumerate(tokens):
            color = COLORS[i % len(COLORS)]
            display = token.replace("Ġ", "·").replace("Ċ", "↵")
            colored.append(f" {display} ", style=f"bold {color} on dark_{color}")
            colored.append(" ")

        console.print(f"\n[bold]Input:[/bold] \"{text}\"")
        console.print(f"[bold]Tokens ({len(tokens)}):[/bold]")
        console.print(colored)

        # Table
        table = Table(title="Token Breakdown")
        table.add_column("#", style="dim")
        table.add_column("Token")
        table.add_column("ID")
        table.add_column("Decoded")

        for i, (token, tid) in enumerate(zip(tokens, token_ids)):
            decoded = self.tokenizer.decode([tid])
            table.add_row(str(i), token, str(tid), repr(decoded))

        console.print(table)
        console.print(f"[dim]Characters: {len(text)} → Tokens: {len(tokens)} "
                     f"(ratio: {len(text)/len(tokens):.1f} chars/token)[/dim]")

    def cmd_compare(self, prompt):
        """Generate multiple outputs at different temperatures."""
        if not prompt:
            prompt = Prompt.ask("[yellow]Enter prompt[/yellow]")
        if not prompt:
            return

        temperatures = [0.3, 0.7, 1.0, 1.5]
        console.print(f"\n[bold]Comparing temperatures for:[/bold] \"{prompt}\"\n")

        table = Table(title="Temperature Comparison")
        table.add_column("Temp", style="bold")
        table.add_column("Generated Text")
        table.add_column("Time", style="dim")

        for temp in temperatures:
            result = self._generate(prompt, temperature=temp)
            table.add_row(
                str(temp),
                result["text"].strip()[:90],
                f"{result['time']:.2f}s",
            )

        console.print(table)

    def cmd_history(self):
        """Show generation history."""
        if not self.history:
            console.print("[dim]No history yet. Generate some text first![/dim]")
            return

        table = Table(title=f"History ({len(self.history)} entries)")
        table.add_column("#", style="dim")
        table.add_column("Prompt")
        table.add_column("Output")
        table.add_column("Model", style="dim")
        table.add_column("Temp", style="dim")

        for i, entry in enumerate(self.history[-10:], 1):  # Last 10
            table.add_row(
                str(i),
                entry["prompt"][:30],
                entry["text"].strip()[:40],
                entry["model"],
                str(entry["params"].get("temperature", "—")),
            )

        console.print(table)

    def run(self):
        """Main application loop."""
        console.print(Panel(
            "[bold cyan]🤖 LLM PLAYGROUND[/bold cyan]\n\n"
            f"Model: [green]{self.model_name}[/green]\n"
            "Type any text to generate, or use commands.\n"
            "Type [bold]/help[/bold] for all commands.",
            title="Welcome",
            border_style="cyan",
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]🤖 >>>[/bold yellow]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            user_input = user_input.strip()

            # Parse commands
            if user_input == "/quit" or user_input == "/exit":
                console.print("[dim]Goodbye! Happy experimenting! 🚀[/dim]")
                break
            elif user_input == "/help":
                console.print(Markdown(HELP_TEXT))
            elif user_input == "/params":
                self.cmd_params()
            elif user_input.startswith("/set "):
                self.cmd_set(user_input[5:])
            elif user_input.startswith("/model"):
                name = user_input[7:].strip() if len(user_input) > 7 else ""
                if not name:
                    name = Prompt.ask("Model name", default="distilgpt2")
                self._load_model(name)
            elif user_input.startswith("/tokens"):
                text = user_input[8:].strip() if len(user_input) > 8 else ""
                self.cmd_tokens(text)
            elif user_input.startswith("/compare"):
                prompt = user_input[9:].strip() if len(user_input) > 9 else ""
                self.cmd_compare(prompt)
            elif user_input == "/history":
                self.cmd_history()
            elif user_input == "/clear":
                self.history.clear()
                console.print("[green]✓ History cleared[/green]")
            elif user_input.startswith("/"):
                console.print(f"[red]Unknown command: {user_input}[/red]")
                console.print("[dim]Type /help for available commands[/dim]")
            else:
                # Treat as a generation prompt
                self.cmd_generate(user_input)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
    playground = FullPlayground(model_name=model)
    playground.run()
```

Run it:

```bash
python playground.py
```

Or with a specific model:

```bash
python playground.py gpt2
```

**Expected output:**
```
╭────────────────── Welcome ──────────────────╮
│ 🤖 LLM PLAYGROUND                          │
│                                             │
│ Model: distilgpt2                           │
│ Type any text to generate, or use commands. │
│ Type /help for all commands.                │
╰─────────────────────────────────────────────╯

🤖 >>> The secret to happiness is

╭────────────── distilgpt2 ──────────────╮
│ The secret to happiness is finding     │
│ what you love and doing it every day   │
│ without hesitation or fear of failure. │
├────────────────────────────────────────┤
│ 23 tokens | 0.84s | 27.4 tok/s        │
╰────────────────────────────────────────╯

🤖 >>> /set temperature 0.3
✓ temperature = 0.3

🤖 >>> /tokens Hello world
Input: "Hello world"
Tokens (2): [Hello] [·world]
Characters: 11 → Tokens: 2 (ratio: 5.5 chars/token)

🤖 >>> /compare The meaning of life is
Comparing temperatures for: "The meaning of life is"

┌─────── Temperature Comparison ───────┐
│ Temp │ Generated Text        │ Time  │
├──────┼───────────────────────┼───────┤
│ 0.3  │ to be happy and...    │ 0.72s │
│ 0.7  │ not just about the... │ 0.68s │
│ 1.0  │ a journey through...  │ 0.71s │
│ 1.5  │ woven into quantum... │ 0.69s │
└──────┴───────────────────────┴───────┘
```

---

## 9. Common Pitfalls & Troubleshooting

### Problem: "No module named 'transformers'"

**Cause:** Dependencies not installed or wrong Python environment.

**Fix:**
```bash
# Make sure virtual environment is active
# Windows:
venv\Scripts\activate
# Then install:
pip install transformers torch rich
```

### Problem: "CUDA out of memory"

**Cause:** Model too large for your GPU memory.

**Fix:**
```python
# Option 1: Use CPU instead
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")

# Option 2: Use a smaller model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Option 3: Use half precision (if you have a GPU)
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
```

### Problem: "This model does not have a pad token"

**Cause:** Some models (like GPT-2) don't define a padding token.

**Fix:**
```python
tokenizer.pad_token = tokenizer.eos_token
```

### Problem: Very slow generation

**Cause:** Running large model on CPU.

**Fixes:**
- Use `distilgpt2` (fastest small model)
- Reduce `max_new_tokens`
- If you have a GPU: `model.to("cuda")`

### Problem: Generated text is repetitive

**Cause:** Temperature too low or no repetition penalty.

**Fix:**
```python
# Add repetition penalty
output = model.generate(
    ...,
    repetition_penalty=1.2,  # Penalize repeating tokens
    no_repeat_ngram_size=3,  # Never repeat 3-word phrases
)
```

### Problem: Generated text is nonsensical

**Cause:** Temperature too high.

**Fix:**
- Lower temperature to 0.7-1.0
- Use top_p=0.9 to filter unlikely tokens
- Use a larger model for better coherence

### Problem: Model download is very slow

**Fix:**
```bash
# Set Hugging Face cache directory to a fast drive
export HF_HOME=/path/to/fast/drive/.cache/huggingface

# Or use the mirror (if available in your region)
export HF_ENDPOINT=https://hf-mirror.com
```

### Problem: "RuntimeError: Expected all tensors on same device"

**Cause:** Model and inputs on different devices (CPU vs GPU).

**Fix:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = tokenizer(text, return_tensors="pt").to(device)
```

---

## Summary of Files

| File | Purpose | Key Concept |
|------|---------|-------------|
| `check_setup.py` | Verify installation | Environment setup |
| `step1_hello.py` | First text generation | Pipeline API |
| `step2_tokenizer.py` | Visualize tokenization | BPE, token IDs |
| `step3_parameters.py` | Compare parameters | Temperature, top-k, top-p |
| `step4_interactive.py` | Interactive playground | Full generation control |
| `step5_compare.py` | Multi-model comparison | Model differences |
| `playground.py` | Full application | Everything combined |

## Next Steps

Once you're comfortable with this playground, try:

1. **Add streaming output** — Show tokens as they're generated (use `TextIteratorStreamer`)
2. **Build a web UI** — Use Gradio or Streamlit for a browser-based interface
3. **Try instruction-tuned models** — Load `microsoft/phi-2` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
4. **Add conversation memory** — Implement a chat-style interface with history
5. **Measure perplexity** — Calculate how "surprised" the model is by different texts

---

*Happy experimenting! 🚀*

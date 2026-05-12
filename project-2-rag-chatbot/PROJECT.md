# Project 2: Build a Customer Support Chatbot

## A Hands-On Tutorial Using RAG and Prompt Engineering

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Step 1: Create Sample Support Documents](#step-1-create-sample-support-documents)
4. [Step 2: Load and Parse Documents](#step-2-load-and-parse-documents)
5. [Step 3: Chunk the Documents](#step-3-chunk-the-documents)
6. [Step 4: Create Embeddings and Index](#step-4-create-embeddings-and-index)
7. [Step 5: Build the Retrieval System](#step-5-build-the-retrieval-system)
8. [Step 6: Build the Generation System](#step-6-build-the-generation-system)
9. [Step 7: Add Prompt Engineering](#step-7-add-prompt-engineering)
10. [Step 8: Build the Complete Chatbot](#step-8-build-the-complete-chatbot)
11. [Step 9: Add Conversation Memory](#step-9-add-conversation-memory)
12. [Step 10: Evaluate Your Chatbot](#step-10-evaluate-your-chatbot)
13. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)

---

## 1. Prerequisites

### Knowledge Required
- Basic Python (variables, functions, lists, dictionaries)
- Command line basics (navigating directories, running scripts)
- No AI/ML knowledge needed — we will explain everything!

### Software Required
- **Python 3.9+** (check with `python --version`)
- **pip** (Python package manager, comes with Python)
- **A text editor** (VS Code recommended)
- **An OpenAI API key** OR **HuggingFace account** (we show both options)

---

## 2. Environment Setup

### Step 2.1: Create Project Directory

```bash
mkdir customer-support-chatbot
cd customer-support-chatbot
```

### Step 2.2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2.3: Install Dependencies

Create a file called `requirements.txt`:

```text
langchain==0.3.25
langchain-community==0.3.24
langchain-huggingface==0.1.2
chromadb==0.6.3
sentence-transformers==3.4.1
huggingface-hub==0.29.3
transformers==4.48.3
torch>=2.0.0
```

Install everything:

```bash
pip install -r requirements.txt
```

> **Note:** If you want to use OpenAI instead of HuggingFace, also install:
> ```bash
> pip install langchain-openai==0.3.12
> ```

### Step 2.4: Verify Installation

Create `verify_setup.py`:

```python
"""Verify that all dependencies are installed correctly."""

def check_imports():
    errors = []

    try:
        import langchain
        print(f"+ langchain {langchain.__version__}")
    except ImportError as e:
        errors.append(f"X langchain: {e}")

    try:
        import chromadb
        print(f"+ chromadb {chromadb.__version__}")
    except ImportError as e:
        errors.append(f"X chromadb: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        print("+ sentence-transformers")
    except ImportError as e:
        errors.append(f"X sentence-transformers: {e}")

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("+ langchain text splitter")
    except ImportError as e:
        errors.append(f"X langchain text splitter: {e}")

    if errors:
        print("\n--- ERRORS ---")
        for err in errors:
            print(err)
        print("\nPlease fix the above errors before continuing.")
    else:
        print("\n+ All dependencies installed successfully!")
        print("You are ready to start building!")

if __name__ == "__main__":
    check_imports()
```

Run it:

```bash
python verify_setup.py
```

**Expected Output:**
```
+ langchain 0.3.25
+ chromadb 0.6.3
+ sentence-transformers
+ langchain text splitter

+ All dependencies installed successfully!
You are ready to start building!
```

---

## Step 1: Create Sample Support Documents

We need documents for our chatbot to learn from. Create `create_docs.py`:

```python
"""Create sample support documents for the chatbot."""
import os

os.makedirs("docs", exist_ok=True)

faq_content = """TechCo Customer Support FAQ

Q: How do I reset my password?
A: To reset your password, go to Settings > Account > Security > Reset Password.
You will receive a verification email within 5 minutes. Click the link in the email
to create a new password. Your new password must be at least 8 characters with one
uppercase letter and one number.

Q: What are your support hours?
A: Our support team is available Monday through Friday, 9 AM to 6 PM Eastern Time.
Weekend support is available via email only, with responses within 24 hours.
Premium plan members have access to 24/7 live chat support.

Q: How do I cancel my subscription?
A: To cancel your subscription, go to Settings > Billing > Subscription > Cancel.
Your access will continue until the end of your current billing period.
No refund is issued for the remaining days. You can reactivate anytime within 30 days
without losing your data.

Q: What payment methods do you accept?
A: We accept Visa, Mastercard, American Express, PayPal, and bank transfers.
All payments are processed securely through Stripe. For enterprise plans,
we also accept purchase orders and wire transfers.

Q: How do I upgrade my plan?
A: Go to Settings > Billing > Change Plan. Select your desired plan and confirm.
The upgrade takes effect immediately. You will be charged the prorated difference
for the remainder of your billing period. All your data and settings are preserved.

Q: Can I get a refund?
A: We offer a full refund within 14 days of purchase for new subscriptions.
After 14 days, refunds are handled on a case-by-case basis. Contact support
with your order number and reason for the refund request. Refunds are processed
within 5-7 business days back to your original payment method.

Q: How do I contact a human agent?
A: You can reach a human agent through: 1) Live chat (click the chat icon, bottom-right),
2) Email: support@techco.com (response within 4 hours), 3) Phone: 1-800-555-0123
(during support hours). Premium members can also use the priority support line.

Q: Is my data secure?
A: Yes! We use AES-256 encryption for data at rest and TLS 1.3 for data in transit.
We are SOC 2 Type II certified and GDPR compliant. We perform regular third-party
security audits. Your data is stored in US-based data centers with 99.99% uptime.
"""

shipping_content = """TechCo Shipping and Delivery Policy

Standard Shipping:
- Delivery time: 5-7 business days
- Cost: Free for orders over $50, otherwise $4.99
- Tracking: Available via email confirmation link

Express Shipping:
- Delivery time: 2-3 business days
- Cost: $12.99
- Tracking: Real-time tracking with SMS updates

Overnight Shipping:
- Delivery time: Next business day (order before 2 PM EST)
- Cost: $24.99
- Tracking: Real-time tracking with SMS and email updates

International Shipping:
- Delivery time: 10-15 business days
- Cost: Varies by destination (calculated at checkout)
- Customs fees: Customer is responsible for any import duties or taxes
- Tracking: Available but may be limited in some regions

Shipping Issues:
- If your package has not arrived within the expected timeframe, please wait 2
  additional business days before contacting support.
- For lost packages, contact us within 30 days of the expected delivery date.
- We will either reship the item or provide a full refund for lost packages.
- Damaged items must be reported within 7 days of delivery with photos of the damage.
"""

product_content = """TechCo Product Guide

Plan Comparison:

Basic Plan ($9.99/month):
- 5 GB storage
- Email support only
- 2 team members max
- Basic analytics
- Standard API access (100 requests/hour)

Pro Plan ($29.99/month):
- 50 GB storage
- Email and chat support
- 10 team members max
- Advanced analytics with export
- Enhanced API access (1000 requests/hour)
- Custom integrations

Enterprise Plan ($99.99/month):
- Unlimited storage
- 24/7 priority support (phone, chat, email)
- Unlimited team members
- Full analytics suite with AI insights
- Unlimited API access
- Custom integrations + dedicated engineer
- SSO and advanced security features
- Custom SLA with 99.99% uptime guarantee

Feature Details:

Storage: Includes all file types (documents, images, videos).
Files up to 5 GB each on Pro and Enterprise. Basic plan limit is 500 MB per file.

Analytics: Basic shows page views and simple metrics.
Advanced (Pro) adds conversion tracking, funnels, and CSV export.
Enterprise adds AI-powered insights and predictive analytics.

API Access: RESTful API with OAuth 2.0 authentication.
Full documentation at docs.techco.com/api.
Rate limits reset every hour. Exceeding limits returns HTTP 429.

Integrations: Pro and Enterprise support Slack, Jira, GitHub,
Salesforce, and 50+ other tools. Enterprise includes custom
integration development with our engineering team.
"""

files = {
    "docs/faq.txt": faq_content,
    "docs/shipping_policy.txt": shipping_content,
    "docs/product_guide.txt": product_content,
}

for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"+ Created {filepath}")

print(f"\n+ All documents created! Total files: {len(files)}")
```

Run it:

```bash
python create_docs.py
```

**Expected Output:**
```
+ Created docs/faq.txt
+ Created docs/shipping_policy.txt
+ Created docs/product_guide.txt

+ All documents created! Total files: 3
```

---

## Step 2: Load and Parse Documents

Create `step2_load_documents.py`:

```python
"""Step 2: Load and parse documents from the docs/ folder."""
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader


def load_documents(docs_dir="docs"):
    """Load all text documents from the docs directory."""
    if not os.path.exists(docs_dir):
        print(f"Error: '{docs_dir}' directory not found! Run create_docs.py first.")
        return []

    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def display_documents(documents):
    """Display information about loaded documents."""
    print(f"Loaded {len(documents)} documents:\n")
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  Document {i+1}: {source}")
        print(f"    Length: {len(doc.page_content)} chars | Preview: {preview}...")
        print()


if __name__ == "__main__":
    print("=" * 50)
    print("Step 2: Loading Documents")
    print("=" * 50)

    documents = load_documents()
    if documents:
        display_documents(documents)
        print(f"+ Successfully loaded {len(documents)} documents!")
    else:
        print("X No documents loaded.")
```

Run: `python step2_load_documents.py`

**Expected Output:**
```
==================================================
Step 2: Loading Documents
==================================================
Loaded 3 documents:

  Document 1: docs\faq.txt
    Length: 1847 chars | Preview: TechCo Customer Support FAQ...

  Document 2: docs\product_guide.txt
    Length: 1205 chars | Preview: TechCo Product Guide...

  Document 3: docs\shipping_policy.txt
    Length: 987 chars | Preview: TechCo Shipping and Delivery Policy...

+ Successfully loaded 3 documents!
```

---

## Step 3: Chunk the Documents

Create `step3_chunk_documents.py`:

```python
"""Step 3: Chunk documents into smaller pieces for indexing."""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from step2_load_documents import load_documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    print("=" * 50)
    print("Step 3: Chunking Documents")
    print("=" * 50)

    documents = load_documents()
    if not documents:
        print("No documents to chunk. Run create_docs.py first.")
        exit(1)

    chunks = chunk_documents(documents)

    print(f"Total chunks created: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]):
        source = chunk.metadata.get("source", "unknown")
        print(f"  Chunk {i+1} ({source}): {len(chunk.page_content)} chars")
        print(f"    '{chunk.page_content[:100]}...'\n")

    sizes = [len(c.page_content) for c in chunks]
    print(f"+ Created {len(chunks)} chunks!")
    print(f"  Sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
```

Run: `python step3_chunk_documents.py`

---

## Step 4: Create Embeddings and Index

Create `step4_create_index.py`:

```python
"""Step 4: Create embeddings and store in ChromaDB vector database."""
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from step2_load_documents import load_documents
from step3_chunk_documents import chunk_documents

CHROMA_DB_DIR = "./chroma_db"


def create_embeddings_model():
    """Create the embedding model (runs locally, no API key needed)."""
    print("Loading embedding model (first time may download ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("+ Embedding model loaded!")
    return embeddings


def create_vector_store(chunks, embeddings, persist_directory=CHROMA_DB_DIR):
    """Create a ChromaDB vector store from document chunks."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    print(f"  Creating vector store with {len(chunks)} chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"+ Vector store saved to {persist_directory}")
    return vector_store


if __name__ == "__main__":
    print("=" * 50)
    print("Step 4: Creating Embeddings and Vector Index")
    print("=" * 50)

    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = create_embeddings_model()
    vector_store = create_vector_store(chunks, embeddings)

    # Test with sample queries
    print("\n--- Testing ---")
    for query in ["How do I reset my password?", "What plans do you offer?", "Shipping time?"]:
        results = vector_store.similarity_search(query, k=2)
        print(f"\n  Query: '{query}'")
        for i, r in enumerate(results):
            print(f"    {i+1}. [{r.metadata.get('source','')}] {r.page_content[:80]}...")

    print(f"\n+ Index created with {len(chunks)} chunks!")
```

Run: `python step4_create_index.py`

**Expected Output:**
```
==================================================
Step 4: Creating Embeddings and Vector Index
==================================================
Loading embedding model (first time may download ~90MB)...
+ Embedding model loaded!
  Creating vector store with 12 chunks...
+ Vector store saved to ./chroma_db

--- Testing ---

  Query: 'How do I reset my password?'
    1. [docs\faq.txt] TechCo Customer Support FAQ  Q: How do I reset my password?...
    2. [docs\faq.txt] Q: How do I contact a human agent?...

+ Index created with 12 chunks!
```

---

## Step 5: Build the Retrieval System

Create `step5_retriever.py`:

```python
"""Step 5: Build the retrieval system to find relevant documents."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DB_DIR = "./chroma_db"


def get_retriever(k=3):
    """Load the vector store and return a retriever."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def retrieve_context(query, retriever):
    """Retrieve relevant document chunks for a query."""
    return retriever.invoke(query)


def format_context(results):
    """Format retrieved results into a context string."""
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    print("=" * 50)
    print("Step 5: Testing Retrieval System")
    print("=" * 50)

    retriever = get_retriever(k=3)
    print("+ Retriever ready!\n")

    for query in ["How can I get a refund?", "What is in the Pro plan?", "My package is late"]:
        print(f"Query: '{query}'")
        results = retrieve_context(query, retriever)
        for i, doc in enumerate(results):
            src = doc.metadata.get("source", "")
            print(f"  {i+1}. [{src}] {doc.page_content[:70]}...")
        print()

    print("+ Retrieval system working!")
```

Run: `python step5_retriever.py`

---

## Step 6: Build the Generation System

Create `step6_generator.py`:

```python
"""Step 6: Build the generation system using an LLM.

Supports:
  - OpenAI (set OPENAI_API_KEY environment variable)
  - HuggingFace (set HUGGINGFACEHUB_API_TOKEN environment variable)
  - Fallback (no API key needed - basic keyword extraction)
"""
import os

USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
USE_HUGGINGFACE = bool(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))


def get_llm():
    """Create and return an LLM instance."""
    if USE_OPENAI:
        from langchain_openai import ChatOpenAI
        print("Using OpenAI GPT model...")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=500)

    elif USE_HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEndpoint
        print("Using HuggingFace model...")
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            temperature=0.3,
            max_new_tokens=500
        )

    else:
        print("No API key found. Using fallback generator.")
        print("  (Set OPENAI_API_KEY or HUGGINGFACEHUB_API_TOKEN for better results)")
        return _get_fallback_llm()


def _get_fallback_llm():
    """Fallback LLM that extracts answers from context without an API."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from typing import List, Optional

    class FallbackLLM(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fallback"

        def _generate(self, messages: List[BaseMessage],
                      stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
            prompt_text = messages[-1].content if messages else ""

            # Extract context and find answer lines
            if "Context" in prompt_text and "Customer Question:" in prompt_text:
                ctx_start = prompt_text.index("Context")
                q_start = prompt_text.index("Customer Question:")
                context = prompt_text[ctx_start:q_start]

                # Extract lines that look like answers
                lines = context.split("\n")
                answer_lines = []
                capturing = False
                for line in lines:
                    if "A:" in line:
                        capturing = True
                        answer_lines.append(line.split("A:", 1)[-1].strip())
                    elif capturing and line.strip() and not line.startswith("Q:") and not line.startswith("["):
                        answer_lines.append(line.strip())
                    elif line.startswith("Q:") or line.startswith("[Source"):
                        capturing = False

                if not answer_lines:
                    # Try extracting bullet points or sentences
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("- ") and len(stripped) > 10:
                            answer_lines.append(stripped)

                if answer_lines:
                    response = "Based on our documentation: " + " ".join(answer_lines[:4])
                else:
                    response = ("I don't have specific information about that. "
                               "Let me connect you with a human agent.")
            else:
                response = "I'd be happy to help! Could you rephrase your question?"

            msg = AIMessage(content=response)
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _identifying_params(self) -> dict:
            return {"model": "fallback"}

    return FallbackLLM()


def generate_answer(llm, context, question):
    """Generate an answer using the LLM with retrieved context."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = """You are a helpful customer support assistant for TechCo.
Answer the customer's question using ONLY the information provided in the Context.
If the answer is not in the context, say "I don't have information about that.
Let me connect you with a human agent."
Be concise, friendly, and helpful."""

    human_prompt = f"""Context:
{context}

Customer Question: {question}

Instructions:
- Answer based only on the context above
- Be concise and direct
- If you are not sure, say so
- Suggest next steps if appropriate

Answer:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    print("=" * 50)
    print("Step 6: Testing Generation System")
    print("=" * 50)

    llm = get_llm()
    print("+ LLM ready!\n")

    sample_context = """[Source: docs/faq.txt]
Q: Can I get a refund?
A: We offer a full refund within 14 days of purchase for new subscriptions.
After 14 days, refunds are handled on a case-by-case basis. Contact support
with your order number and reason for the refund request. Refunds are processed
within 5-7 business days back to your original payment method."""

    question = "I want a refund. I bought my subscription 10 days ago."

    print(f"Question: {question}")
    print("-" * 40)
    answer = generate_answer(llm, sample_context, question)
    print(f"Answer: {answer}")
    print("\n+ Generation system working!")
```

Run: `python step6_generator.py`

**Expected Output:**
```
==================================================
Step 6: Testing Generation System
==================================================
No API key found. Using fallback generator.
+ LLM ready!

Question: I want a refund. I bought my subscription 10 days ago.
----------------------------------------
Answer: Based on our documentation: We offer a full refund within 14 days of
purchase for new subscriptions. After 14 days, refunds are handled on a
case-by-case basis. Contact support with your order number and reason for
the refund request.

+ Generation system working!
```

---

## Step 7: Add Prompt Engineering

Create `step7_prompts.py`:

```python
"""Step 7: Advanced prompt engineering for better chatbot responses."""


class PromptTemplates:
    """Collection of prompt templates for different scenarios."""

    SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable customer support
assistant for TechCo. You have the following characteristics:

Personality:
- Warm and empathetic - acknowledge the customer's situation
- Professional but conversational - avoid corporate jargon
- Solution-oriented - always provide actionable next steps
- Honest - if you don't know something, say so

Rules:
- ONLY use information from the provided Context
- NEVER make up information that is not in the Context
- If the answer is not in the Context, offer to connect with a human agent
- Keep responses concise (2-4 sentences unless detail is needed)
- Always end with a helpful follow-up question or next step"""

    RAG_TEMPLATE = """Context (from our knowledge base):
---
{context}
---

Customer Question: {question}

Instructions:
1. Answer ONLY based on the Context above
2. If the Context does not contain the answer, say: "I don't have specific
   information about that. Let me connect you with a human agent."
3. Be friendly and match the customer's tone
4. Provide step-by-step instructions when applicable
5. End with a follow-up question or next step

Answer:"""

    FEW_SHOT_TEMPLATE = """Here are examples of ideal responses:

Example 1:
Customer: "How do I change my email?"
Alex: "To update your email address, head to Settings > Account > Profile and
click 'Edit Email.' You'll need to verify the new email address before the change
takes effect. Is there anything else I can help with?"

Example 2:
Customer: "This is so frustrating, nothing works!"
Alex: "I'm really sorry you're having trouble - I understand how frustrating that
can be. Let's figure this out together. Could you tell me specifically what's not
working? For example, is it related to logging in, a feature, or billing?"

Example 3:
Customer: "What's the cheapest plan?"
Alex: "Our most affordable option is the Basic Plan at $9.99/month, which includes
5 GB storage, email support, and access for up to 2 team members. Would you like
me to compare it with our other plans to find the best fit?"

---
Now respond to this customer using the context below:

Context:
{context}

Customer: {question}

Alex:"""

    CHAIN_OF_THOUGHT_TEMPLATE = """Context:
{context}

Customer Question: {question}

Let me think through this step by step:
1. What is the customer asking about?
2. What relevant information is in the context?
3. What is the best answer based on that information?
4. What follow-up might be helpful?

Based on my analysis, here is my response:"""


def get_prompt(template_name, context, question):
    """Get a formatted prompt by template name."""
    templates = {
        "rag": PromptTemplates.RAG_TEMPLATE,
        "few_shot": PromptTemplates.FEW_SHOT_TEMPLATE,
        "chain_of_thought": PromptTemplates.CHAIN_OF_THOUGHT_TEMPLATE,
    }

    template = templates.get(template_name, PromptTemplates.RAG_TEMPLATE)
    return template.format(context=context, question=question)


if __name__ == "__main__":
    print("=" * 50)
    print("Step 7: Prompt Engineering Templates")
    print("=" * 50)

    sample_context = "Pro Plan ($29.99/month): 50 GB storage, email and chat support, 10 team members"
    sample_question = "How much storage do I get with Pro?"

    print("\n--- RAG Template ---")
    print(get_prompt("rag", sample_context, sample_question)[:300])

    print("\n--- Few-Shot Template ---")
    print(get_prompt("few_shot", sample_context, sample_question)[:500])

    print("\n--- Chain-of-Thought Template ---")
    print(get_prompt("chain_of_thought", sample_context, sample_question)[:300])

    print("\n+ All prompt templates ready!")
```

Run: `python step7_prompts.py`

---

## Step 8: Build the Complete Chatbot

Now we combine everything into a working chatbot. Create `chatbot.py`:

```python
"""Step 8: Complete Customer Support Chatbot with RAG."""
from step5_retriever import get_retriever, retrieve_context, format_context
from step6_generator import get_llm, generate_answer
from step7_prompts import PromptTemplates, get_prompt
from langchain_core.messages import HumanMessage, SystemMessage


class CustomerSupportChatbot:
    """A RAG-powered customer support chatbot."""

    def __init__(self, prompt_style="few_shot"):
        """Initialize the chatbot.

        Args:
            prompt_style: One of "rag", "few_shot", or "chain_of_thought"
        """
        print("Initializing chatbot...")
        self.retriever = get_retriever(k=3)
        self.llm = get_llm()
        self.prompt_style = prompt_style
        self.conversation_history = []
        print("+ Chatbot ready!\n")

    def ask(self, question):
        """Ask the chatbot a question and get a response.

        Args:
            question: The customer's question

        Returns:
            The chatbot's response string
        """
        # Step 1: Retrieve relevant context
        results = retrieve_context(question, self.retriever)
        context = format_context(results)

        # Step 2: Build the prompt using our templates
        user_prompt = get_prompt(self.prompt_style, context, question)

        # Step 3: Generate the answer
        messages = [
            SystemMessage(content=PromptTemplates.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        response = self.llm.invoke(messages)
        answer = response.content

        # Step 4: Store in conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": [r.metadata.get("source", "") for r in results]
        })

        return answer

    def get_sources(self):
        """Get the sources used in the last response."""
        if self.conversation_history:
            return self.conversation_history[-1]["sources"]
        return []


def run_interactive():
    """Run the chatbot in interactive mode."""
    print("=" * 60)
    print("    TechCo Customer Support Chatbot")
    print("    Type 'quit' to exit, 'sources' to see last sources")
    print("=" * 60)

    bot = CustomerSupportChatbot(prompt_style="few_shot")

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nThank you for chatting! Have a great day!")
            break
        if question.lower() == "sources":
            sources = bot.get_sources()
            print(f"Sources used: {sources}")
            continue

        answer = bot.ask(question)
        print(f"\nAlex: {answer}")


if __name__ == "__main__":
    run_interactive()
```

Run: `python chatbot.py`

**Expected Interaction:**
```
============================================================
    TechCo Customer Support Chatbot
    Type 'quit' to exit, 'sources' to see last sources
============================================================
Initializing chatbot...
+ Chatbot ready!

You: How do I reset my password?

Alex: To reset your password, go to Settings > Account > Security > Reset Password.
You'll receive a verification email within 5 minutes - just click the link to create
your new password. Make sure it's at least 8 characters with one uppercase letter and
one number. Need help with anything else?

You: What plans do you have?

Alex: We offer three plans: Basic ($9.99/month) with 5 GB storage, Pro ($29.99/month)
with 50 GB storage and chat support, and Enterprise ($99.99/month) with unlimited
everything plus priority support. Would you like me to help you choose the right one?

You: quit

Thank you for chatting! Have a great day!
```

---

## Step 9: Add Conversation Memory

Create `step9_memory_chatbot.py` to add multi-turn conversation awareness:

```python
"""Step 9: Chatbot with conversation memory for multi-turn support."""
from step5_retriever import get_retriever, retrieve_context, format_context
from step6_generator import get_llm
from step7_prompts import PromptTemplates
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class MemoryChatbot:
    """Chatbot that remembers conversation history for follow-up questions."""

    def __init__(self, max_history=5):
        """Initialize with conversation memory.

        Args:
            max_history: Maximum number of turns to remember
        """
        print("Initializing chatbot with memory...")
        self.retriever = get_retriever(k=3)
        self.llm = get_llm()
        self.max_history = max_history
        self.messages = [SystemMessage(content=self._system_prompt())]
        print("+ Chatbot with memory ready!\n")

    def _system_prompt(self):
        return """You are Alex, a friendly customer support assistant for TechCo.

Rules:
- Answer ONLY using the provided Context
- If the answer is not in the Context, say so and offer to connect with a human
- Remember the conversation history to understand follow-up questions
- Be concise, friendly, and helpful
- Always suggest a next step"""

    def ask(self, question):
        """Process a question with full conversation context."""
        # Retrieve relevant context
        results = retrieve_context(question, self.retriever)
        context = format_context(results)

        # Build message with context
        user_message = f"""Context from knowledge base:
---
{context}
---

Customer says: {question}"""

        self.messages.append(HumanMessage(content=user_message))

        # Trim history if too long (keep system prompt + last N exchanges)
        if len(self.messages) > (self.max_history * 2 + 1):
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]

        # Generate response
        response = self.llm.invoke(self.messages)
        answer = response.content

        # Store assistant response in history
        self.messages.append(AIMessage(content=answer))

        return answer

    def reset(self):
        """Clear conversation history."""
        self.messages = [SystemMessage(content=self._system_prompt())]
        print("Conversation history cleared.")


def run_interactive():
    """Run the memory chatbot interactively."""
    print("=" * 60)
    print("    TechCo Support Chatbot (with Memory)")
    print("    Commands: 'quit', 'reset', 'history'")
    print("=" * 60)

    bot = MemoryChatbot(max_history=5)

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nThank you! Have a great day!")
            break
        if question.lower() == "reset":
            bot.reset()
            continue
        if question.lower() == "history":
            print(f"  Conversation turns: {(len(bot.messages) - 1) // 2}")
            continue

        answer = bot.ask(question)
        print(f"\nAlex: {answer}")


if __name__ == "__main__":
    run_interactive()
```

Run: `python step9_memory_chatbot.py`

**Expected Interaction (showing memory):**
```
You: What's your cheapest plan?

Alex: Our Basic Plan is $9.99/month with 5 GB storage, email support, and
up to 2 team members. Would you like to know more about what's included?

You: How much storage does it have?

Alex: The Basic Plan includes 5 GB of storage with a 500 MB per-file limit.
If you need more, our Pro Plan offers 50 GB. Would that work better for you?

You: What about the Pro?

Alex: The Pro Plan is $29.99/month and includes 50 GB storage, email and chat
support, up to 10 team members, advanced analytics with export, and 1000 API
requests/hour. It also supports custom integrations. Want me to help you upgrade?
```

---

## Step 10: Evaluate Your Chatbot

Create `step10_evaluate.py` to measure chatbot quality:

```python
"""Step 10: Evaluate the chatbot's response quality."""
from step5_retriever import get_retriever, retrieve_context, format_context
from step6_generator import get_llm, generate_answer


# Test cases: (question, expected_keywords, expected_source)
TEST_CASES = [
    {
        "question": "How do I reset my password?",
        "expected_keywords": ["settings", "account", "security", "email", "8 characters"],
        "expected_source": "faq.txt",
    },
    {
        "question": "What's the Pro plan price?",
        "expected_keywords": ["29.99", "50 GB", "10 team"],
        "expected_source": "product_guide.txt",
    },
    {
        "question": "How long is standard shipping?",
        "expected_keywords": ["5-7", "business days"],
        "expected_source": "shipping_policy.txt",
    },
    {
        "question": "Can I get a refund after 20 days?",
        "expected_keywords": ["14 days", "case-by-case"],
        "expected_source": "faq.txt",
    },
    {
        "question": "What encryption do you use?",
        "expected_keywords": ["AES-256", "TLS"],
        "expected_source": "faq.txt",
    },
]


def evaluate_context_relevance(results, expected_source):
    """Check if retrieved documents are from the expected source."""
    sources = [r.metadata.get("source", "") for r in results]
    relevant = sum(1 for s in sources if expected_source in s)
    return relevant / len(sources) if sources else 0


def evaluate_faithfulness(answer, context):
    """Check if the answer contains info that IS in the context."""
    # Simple check: are answer sentences grounded in context?
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = answer_words & context_words
    # Remove common stop words for better signal
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                  "for", "on", "with", "and", "or", "your", "you", "i", "we", "our"}
    meaningful_overlap = overlap - stop_words
    meaningful_answer = answer_words - stop_words
    if not meaningful_answer:
        return 0
    return len(meaningful_overlap) / len(meaningful_answer)


def evaluate_answer_correctness(answer, expected_keywords):
    """Check if the answer contains expected keywords."""
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(found) / len(expected_keywords) if expected_keywords else 0


def run_evaluation():
    """Run full evaluation suite."""
    print("=" * 60)
    print("    Chatbot Evaluation")
    print("=" * 60)

    retriever = get_retriever(k=3)
    llm = get_llm()

    total_relevance = 0
    total_faithfulness = 0
    total_correctness = 0

    for i, test in enumerate(TEST_CASES):
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        expected_source = test["expected_source"]

        print(f"\nTest {i+1}: '{question}'")
        print("-" * 50)

        # Retrieve
        results = retrieve_context(question, retriever)
        context = format_context(results)

        # Generate
        answer = generate_answer(llm, context, question)

        # Evaluate
        relevance = evaluate_context_relevance(results, expected_source)
        faithfulness = evaluate_faithfulness(answer, context)
        correctness = evaluate_answer_correctness(answer, expected_keywords)

        total_relevance += relevance
        total_faithfulness += faithfulness
        total_correctness += correctness

        print(f"  Answer: {answer[:150]}...")
        print(f"  Context Relevance:  {relevance:.2f}")
        print(f"  Faithfulness:       {faithfulness:.2f}")
        print(f"  Answer Correctness: {correctness:.2f}")
        found_kw = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        missed_kw = [kw for kw in expected_keywords if kw.lower() not in answer.lower()]
        if found_kw:
            print(f"  Found keywords: {found_kw}")
        if missed_kw:
            print(f"  Missed keywords: {missed_kw}")

    n = len(TEST_CASES)
    print("\n" + "=" * 60)
    print("    OVERALL SCORES")
    print("=" * 60)
    print(f"  Context Relevance:  {total_relevance/n:.2f}")
    print(f"  Faithfulness:       {total_faithfulness/n:.2f}")
    print(f"  Answer Correctness: {total_correctness/n:.2f}")
    print(f"  Overall Average:    {(total_relevance + total_faithfulness + total_correctness) / (3*n):.2f}")
    print()

    avg = (total_relevance + total_faithfulness + total_correctness) / (3 * n)
    if avg > 0.7:
        print("  EXCELLENT! Your chatbot is performing well.")
    elif avg > 0.5:
        print("  GOOD. Consider tuning chunk size or prompt templates.")
    else:
        print("  NEEDS WORK. Try adjusting retrieval k, chunk overlap, or prompts.")


if __name__ == "__main__":
    run_evaluation()
```

Run: `python step10_evaluate.py`

**Expected Output:**
```
============================================================
    Chatbot Evaluation
============================================================

Test 1: 'How do I reset my password?'
--------------------------------------------------
  Answer: Based on our documentation: To reset your password, go to Settings...
  Context Relevance:  0.67
  Faithfulness:       0.45
  Answer Correctness: 0.80
  Found keywords: ['settings', 'account', 'security', 'email']
  Missed keywords: ['8 characters']

...

============================================================
    OVERALL SCORES
============================================================
  Context Relevance:  0.60
  Faithfulness:       0.42
  Answer Correctness: 0.72
  Overall Average:    0.58

  GOOD. Consider tuning chunk size or prompt templates.
```

---

## Common Pitfalls and Troubleshooting

### Installation Issues

| Problem | Solution |
|---------|----------|
| `pip install` fails with permissions | Use `pip install --user` or activate your venv |
| `torch` install is huge/slow | Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only |
| Import errors after install | Make sure your venv is activated |
| `chromadb` build errors on Windows | Install Visual C++ Build Tools first |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| "No module named step2_load_documents" | Run scripts from the project root directory |
| ChromaDB "database not found" | Run `step4_create_index.py` first to create the index |
| First run is very slow | The embedding model downloads ~90MB on first use; subsequent runs are fast |
| Out of memory errors | Reduce `chunk_size` or process fewer documents at once |

### Quality Issues

| Problem | Solution |
|---------|----------|
| Chatbot gives irrelevant answers | Increase retrieval `k` value (try k=5) |
| Answers are too generic | Decrease `chunk_size` (try 300) for more focused chunks |
| Chatbot "halluccinates" | Strengthen the "ONLY use context" instruction in prompts |
| Missing important context | Increase `chunk_overlap` (try 100) to avoid splitting key info |
| Answers repeat the question | Add "Do not repeat the question" to your prompt |

### Tuning Guide

```
Problem: Low Context Relevance
  -> Try: chunk_size=300, chunk_overlap=100
  -> Try: k=5 (retrieve more chunks)
  -> Try: A different embedding model (e.g., "all-mpnet-base-v2")

Problem: Low Faithfulness
  -> Try: Stronger system prompt ("cite your sources")
  -> Try: Adding "If unsure, say I don't know" to prompt
  -> Try: Lower LLM temperature (0.1 instead of 0.3)

Problem: Low Answer Correctness
  -> Try: Few-shot prompting (show examples of good answers)
  -> Try: Chain-of-thought prompting for complex questions
  -> Try: A more capable LLM model
```

### API Key Setup

**HuggingFace (Free):**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Set it:
   ```bash
   # Windows
   set HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   # macOS/Linux
   export HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   ```

**OpenAI (Paid):**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Set it:
   ```bash
   # Windows
   set OPENAI_API_KEY=sk-your_key_here
   # macOS/Linux
   export OPENAI_API_KEY=sk-your_key_here
   ```

---

## Project Structure (Final)

```
customer-support-chatbot/
|-- requirements.txt          # Dependencies
|-- verify_setup.py           # Installation checker
|-- create_docs.py            # Creates sample documents
|-- step2_load_documents.py   # Document loading
|-- step3_chunk_documents.py  # Text chunking
|-- step4_create_index.py     # Embeddings + ChromaDB
|-- step5_retriever.py        # Retrieval system
|-- step6_generator.py        # LLM generation
|-- step7_prompts.py          # Prompt templates
|-- chatbot.py                # Complete chatbot (Step 8)
|-- step9_memory_chatbot.py   # Chatbot with memory
|-- step10_evaluate.py        # Evaluation suite
|-- docs/                     # Knowledge base documents
|   |-- faq.txt
|   |-- shipping_policy.txt
|   +-- product_guide.txt
+-- chroma_db/                # Vector database (auto-generated)
```

---

## Next Steps

Once you have the basic chatbot working, try these enhancements:

1. **Add more documents** - PDF support with `PyPDFLoader`, web pages with `WebBaseLoader`
2. **Improve retrieval** - Try hybrid search (keyword + vector), or reranking
3. **Better prompts** - Experiment with different prompt styles for your use case
4. **Add a web UI** - Use Gradio or Streamlit for a visual interface
5. **Deploy** - Package as a Docker container or deploy to a cloud service
6. **Production LLM** - Switch from fallback to OpenAI or a self-hosted model

---

*Refer back to [README.md](./README.md) for the theory behind each concept used in this project.*

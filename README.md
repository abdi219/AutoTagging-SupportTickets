# Auto Tagging Support Tickets Using LLMs

An end-to-end customer support ticket classification pipeline developed during the **AI/ML Engineering Internship at DeveloperHub**. This project demonstrates how an instruction-tuned Large Language Model (LLM) can automatically classify unstructured customer support tickets into ranked operational categories while comparing the effectiveness of **Zero-Shot** and **Few-Shot Prompt Engineering**.

---

## Project Overview

Customer support teams receive thousands of free-form tickets every day. Manually categorizing these tickets is time-consuming and prone to inconsistencies.

This project automates the classification process by predicting the **Top 3 most relevant categories** for every incoming ticket.

Supported categories include:

- Billing
- Technical Support
- Account Security
- Performance

The repository also evaluates how prompt engineering affects model reliability, output formatting, and classification quality.

---

# Features

- Automated support ticket classification
- Zero-Shot and Few-Shot prompt comparison
- Ranked Top-3 category prediction
- Production-friendly prompt templates
- Token-level output stopping
- CSV export for downstream systems
- GPU-accelerated inference using Google Colab

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Language Model | Qwen2.5-1.5B-Instruct |
| Framework | Hugging Face Transformers |
| Device Management | Accelerate |
| Environment | Google Colab |
| Hardware | NVIDIA T4 GPU |
| Data Processing | Pandas, NumPy |
| Language | Python |

---

# Project Workflow

```text
Raw Support Tickets
        │
        ▼
Prompt Construction
        │
        ▼
Zero-Shot / Few-Shot Prompt
        │
        ▼
Qwen2.5-1.5B-Instruct
        │
        ▼
Top 3 Ranked Categories
        │
        ▼
Pandas DataFrame
        │
        ▼
tagged_support_tickets_evaluation.csv
```

---

# Prompt Engineering

## Zero-Shot Prompting

The model receives:

- Customer support ticket
- Allowed category list
- Formatting instructions

No examples are provided.

```python
def get_zero_shot_prompt(ticket_text):
    return f"""
You are an expert customer support agent system.

Allowed Categories:
Billing
Technical Support
Account Security
Performance

Ticket:
"{ticket_text}"

Output exactly the top 3 most probable categories ranked from highest to lowest probability.

Format:
1. Category
2. Category
3. Category

Do not add explanations.

Top 3 Tags:
"""
```

---

## Few-Shot Prompting

Few-shot prompting introduces labeled examples before inference to teach the model the expected behavior.

```python
def get_few_shot_prompt(ticket_text):
    return f"""
Example 1

Ticket:
"I cannot log into my account."

Top 3 Tags:
1. Account Security
2. Technical Support
3. Performance

Example 2

Ticket:
"I need a copy of my invoice."

Top 3 Tags:
1. Billing
2. Account Security
3. Technical Support

Now classify:

"{ticket_text}"

Top 3 Tags:
"""
```

---

# Production Guardrails

Instruction-tuned language models often continue generating explanations even after producing the desired output.

To enforce deterministic responses, the inference pipeline uses tokenizer-aware stopping strings.

```python
pipeline_tokenizer = classifier_llm.tokenizer

classifier_llm(
    prompt,
    max_new_tokens=40,
    return_full_text=False,
    tokenizer=pipeline_tokenizer,
    stop_strings=[
        "\n",
        "Explanation:",
        "This ticket"
    ]
)
```

This prevents unnecessary text generation and ensures every prediction ends immediately after the third ranked category.

---

# Evaluation Results

| Ticket | Customer Ticket | Zero-Shot | Few-Shot |
|:------:|-----------------|-----------|----------|
| 101 | Charged twice for my subscription | Billing, Technical Support, Performance | 1. Billing, 2. Technical Support, 3. Account Security |
| 102 | Desktop app opens to a black screen | Technical Support, Account Security, Billing | 1. Performance, 2. Technical Support, 3. Account Security |
| 103 | Password reset email never arrives | Account Security, Technical Support, Billing | 1. Account Security, 2. Technical Support, 3. Billing |
| 104 | Application freezes while importing CSV files | Technical Support, Performance, Account Security | 1. Performance, 2. Technical Support, 3. Billing |
| 105 | Billing page loads forever while updating payment card | Technical Support, Billing, Account Security | 1. Technical Support, 2. Billing, 3. Account Security |
| 106 | Account locked after failed login attempts | Performance, Technical Support, Account Security | 1. Account Security, 2. Technical Support, 3. Performance |
| 107 | Application crashed while exporting analytics | Application Issues, System Crash, Data Export | 1. Technical Support, 2. Performance, 3. Billing |

---

# Analysis

## Zero-Shot Performance

### Formatting Drift

The model frequently ignored the requested output structure and generated conversational responses.

### Schema Violations

Instead of restricting itself to the allowed categories, the model occasionally hallucinated new labels.

Example:

```text
Application Issues
System Crash
Data Export
```

### Semantic Errors

Some security-related tickets were incorrectly prioritized as performance problems.

---

## Few-Shot Performance

### Consistent Formatting

Every prediction followed the required structure.

```text
1. Category
2. Category
3. Category
```

### No Label Hallucination

The model remained within the predefined category list across every evaluated ticket.

### Better Context Understanding

Few-shot prompting correctly prioritized:

- Security incidents
- Billing issues
- Technical failures

while maintaining consistent ranking behavior.

---

# Output

After inference, predictions are stored in a Pandas DataFrame and exported as:

```python
df.to_csv(
    "tagged_support_tickets_evaluation.csv",
    index=False
)
```

The generated CSV contains:

- Original ticket
- Zero-shot prediction
- Few-shot prediction

making it suitable for analytics dashboards, databases, and downstream ML pipelines.

---

# Repository Structure

```text
.
├── Auto_Tagging_Support_Tickets.ipynb
├── tagged_support_tickets_evaluation.csv
├── README.md
└── requirements.txt
```

---

# Future Improvements

- Fine-tune on real customer support datasets
- Add confidence scores
- Multi-label probability estimation
- FastAPI deployment
- Docker containerization
- REST API endpoint
- Streamlit dashboard
- Batch inference optimization

---

# Key Takeaways

| Zero-Shot | Few-Shot |
|------------|-----------|
| Formatting drift | Stable formatting |
| Hallucinated labels | Restricted categories |
| Less consistent | Highly consistent |
| Higher token usage | Lower token usage |
| Lower reliability | Better production readiness |

---

# Author

Developed as part of the **AI/ML Engineering Internship at DeveloperHub**.

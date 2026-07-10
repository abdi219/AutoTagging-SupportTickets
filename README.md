# Auto Tagging Support Tickets Using LLM

This project was developed as part of the AI/ML Engineering Internship at DeveloperHub. It implements an automated support ticket classification pipeline that uses an instruction-tuned Large Language Model to categorize raw customer service text into ranked tags and outputs the top 3 most probable categories for each ticket.

## Project Objective
The goal is to automatically organize unstructured text data from incoming customer support tickets into defined operational categories: Billing, Technical Support, Account Security, and Performance. The system evaluates the performance difference between zero-shot inference and few-shot learning techniques to establish a stable deployment configuration.

## System Architecture and Tech Stack
* Language Model: Qwen2.5-1.5B-Instruct (via Hugging Face Transformers API)
* Execution Environment: Google Colab with an active NVIDIA T4 GPU accelerator
* Core Libraries: Transformers, Accelerate, Pandas, NumPy

## Prompt Engineering Methodologies

### Zero-Shot Learning Configuration
The model receives the raw customer support ticket text along with strict instructions outlining the allowed categories. It is tasked with identifying and ranking the top 3 categories based purely on its pre-trained semantic weights without seeing any practical target examples.

### Few-Shot Learning Configuration
The model prompt is updated to include explicit example pairs. These examples demonstrate exactly how past tickets were processed, how the categories should be prioritized based on context, and how the final output string must be structured.

## Analytical Observations and Findings

### Zero-Shot Analysis
* Formatting Drift: Without explicit example references, the base model naturally defaults to conversational dialogue. It tends to append long explanations, justifications, and introductory filler text.
* Structural Issues: The model struggles to maintain a strict format. For example, on Ticket 101, it returned unformatted index mappings ("2, 0") before printing categories, and on Ticket 103, it included meta-instructions about ties in the output text.
* Token Budget Extraction: Because the model wastes its token quota printing conversational sentences, the actual classification strings run a high risk of cutting off mid-sentence.

### Few-Shot Analysis
* Behavioral Alignment: Providing just two context examples successfully overrides the model's natural conversational tendency, forcing it to act as a strict data-extraction engine.
* Structural Integrity: The output across all tickets remains perfectly uniform, successfully matching the requested design template: "1. Category, 2. Category, 3. Category".
* High Efficiency: By skipping analytical descriptions and jumping straight to the classification array, the model uses fewer tokens and completes its tasks much faster.

## Results and Performance Comparison

| Ticket ID | Raw Input Text | Zero-Shot Output | Few-Shot Output |
| :--- | :--- | :--- | :--- |
| **101** | Charged twice for monthly subscription. Please refund. | Billing, Account Security, Performance Explanation... | 1. Billing, 2. Technical Support, 3. Account Security |
| **102** | Screen goes completely black when trying to open desktop app. | Performance, Technical Support, Account Security Explanation... | 1. Technical Support, 2. Performance, 3. Account Security |
| **103** | Forgot password and reset link email is not arriving. | 1. Account Security, 2. Technical Support, 3. Billing... | 1. Account Security, 2. Technical Support, 3. Billing |
| **104** | Application lagging terribly when importing CSV files. | Performance, Technical Support, Account Security This category... | 1. Performance, 2. Technical Support, 3. Billing |

## Pipeline Execution and Serialization
The end-to-end pipeline processes incoming arrays, wraps them into a structured Pandas DataFrame, runs parallel inference calls through the loaded GPU model, and outputs a physical production asset named `tagged_support_tickets_evaluation.csv`. This output file is ready to be parsed directly by downstream database management systems or business intelligence tools.

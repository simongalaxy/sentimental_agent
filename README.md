# Amazon Product Review - Sentiment Analysis using Hugging Face and Ollama LLM

A fully automated review analysis pipeline powered by models from Hugging Face and Ollama. It loads and cleans Amazon product review data, removes duplicates, consolidates reviews with similar semantic meanings, extracts sentiments and categories from review texts, labels each record accordingly, and generates product- and platform-specific summaries and insights in Markdown format.

## Motivation

Categorizing, consolidating, and analyzing large volumes of customer reviews in a structured manner is time-consuming, difficult, and subjective when done manually. While Large Language Models (LLMs) can automate this process, relying solely on them is slow, token-intensive, and often yields unpredictable outputs.

---

## How This Project Solves These Problems

Given the high volume of reviews—many of which are exact duplicates or semantically similar—processing every single review through an LLM is not a viable solution. Instead, this pipeline combines specialized tools to enhance speed, improve accuracy, and minimize token usage:

1. **Hugging Face Models for *Review Data Preprocessing*:**
   - Deduplicate and consolidate reviews with identical or similar semantic meanings.
   - Extract sentiment scores from review texts.
   - Automatically generate categories by product and platform, then label each review text (**Fully Automated Categorization**).
2. **Ollama Large Language Model for *Generating Summaries and Insights*:**
   - Extract actionable insights from preprocessed DataFrames grouped by product and platform.
   - Generate concise summaries for both positive and negative reviews.
3. **Pandas for *Producing Statistics and Graphs*:**
   - Aggregate metrics and visualize review distributions.

---

## How the *Pipeline* Works

1. **Data Processing:** Load the CSV file, clean the raw data, and generate a list of DataFrames grouped by product and platform using **Pandas**.
2. **Deduplication and Consolidation:** Identify and merge exact or semantically similar reviews using **SentenceTransformer**, `paraphrase_mining`, and **Pandas**.
3. **Sentiment Analysis:** Classify the sentiment of each review using a **Hugging Face Transformers Pipeline**.
4. **Categorization and Labeling:** Dynamically discover review categories and assign labels using **Text Generation** and **Zero-Shot Classification** pipelines in Hugging Face Transformers.
5. **Statistics and Chart Generation:** Generate summary statistics and visual charts using **Pandas**.
6. **Insight Generation and Summarization:** Produce comprehensive insights and summaries for positive and negative reviews using an **Ollama LLM**.

By handling heavy preprocessing with specialized Hugging Face models, the pipeline runs significantly faster and more accurately. The saved tokens are then dedicated entirely to generating high-value insights and summaries via the Ollama LLM.

---

## Project Structure

*(Add your directory tree or file descriptions here)*

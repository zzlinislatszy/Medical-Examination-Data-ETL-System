# Medical Examination Data ETL System
This project is an end-to-end data processing system that transforms semi-structured JSON inputs into cleaned, structured, and hierarchical text outputs, with optional LLM integration for text refinement.

## Flow
The pipeline consists of the following stages:

### 1. Data Ingestion & Mapping
- Accepts JSON-formatted input via API or function call
- Expands nested structures into tabular form using pandas
- Enriches input records through MongoDB queries (via pymongo)
- Standardizes fields into a unified DataFrame schema

Goal: Convert semi-structured input into a structured, analysis-ready format.

### 2. Data Cleaning & Preprocessing
- Removes duplicate records based on key field combinations
- Normalizes text fields (newline removal, whitespace cleanup, full-width to half-width conversion)
- Handles missing values with configurable defaults
- Sorts records to preserve logical and display order

Goal: Ensure data quality, consistency, and deterministic downstream processing.

### 3. Hierarchical Data Transformation
- Groups records into multi-level structures (e.g., Group → Item → Detail → Summary)
- Preserves original ordering while aggregating related records
- Converts structured data into readable, hierarchical text blocks

Goal: Transform flat tabular data into structured narrative-style output.

### 4. LLM Integration (Optional)
- Integrates an LLM interface to refine or rewrite summary text
- Supports batch processing with concurrency and retry handling
- Provides a mock fallback mode when no API credentials are configured

Goal: Demonstrate how LLMs can be safely integrated into data pipelines without coupling core logic to model availability.

### 5. Output & Delivery
- Outputs intermediate and final results as CSV files
- Returns processed text results as JSON via API
- Designed to be reproducible and suitable for both testing and production-like workflows

## Project Structure
```
Medical Examination Data ETL System/
├── app.py                       # FastAPI entry point
├── db_to_dataframe.py           # JSON -> DataFrame + (optional) pymongo enrichment via env vars
├── data_preprocessing.py        # data cleaning / normalization
├── text_processing.py           # hierarchical text generation API
├── llm_processing.py            # LLM interface (supports mock mode when no keys)
└── utils.py                     # shared utilities
```

## Run
```bash
pip install -r requirements.txt
python app.py
```

Open:
- `GET /` health check
- `POST /process` to process input

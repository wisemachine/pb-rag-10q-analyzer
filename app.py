import os
import json
import re
import spacy

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()
DB_FOLDER = "chroma_db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# 1. Summarization Model (Flan-T5)
# -------------------------------------------------------------------
def initialize_summarizer_model():
    print("Initializing summarization model...")
    model_name = "google/flan-t5-base"  # Or flan-t5-large if you have memory & want fewer token issues
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer_pipeline = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        max_length=150,
        min_length=50
    )
    summarizer = HuggingFacePipeline(pipeline=summarizer_pipeline)
    print("Summarization model loaded successfully!")
    return summarizer

# -------------------------------------------------------------------
# 2. Vector Store (optional RAG usage)
# -------------------------------------------------------------------
def initialize_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    vector_store = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)
    return vector_store, embedding_model

def add_documents_to_store(vector_store, documents):
    texts = [doc.page_content for doc in documents]
    metadatas = [{"page_number": i} for i in range(len(documents))]
    vector_store.add_texts(texts, metadatas=metadatas)
    vector_store.persist()  # persist to disk

# -------------------------------------------------------------------
# 3. Process the 10-Q PDF: load -> chunk -> summarize
# -------------------------------------------------------------------
def process_10q(ticker, summarizer):
    folder_path = os.path.join("10q_docs", ticker)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"No folder found for ticker: {ticker}")
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in folder: {folder_path}")
    pdf_path = os.path.join(folder_path, pdf_files[0])

    print(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Summarize with chunk approach
    full_text = "\n".join([doc.page_content for doc in documents])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)

    print("Summarizing the full document in chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        try:
            summary = summarizer(chunk)
            if summary:
                chunk_summaries.append(summary)
        except Exception as e:
            print(f"Failed to summarize chunk {i + 1}: {e}")
            continue

    consolidated_summary = " ".join(chunk_summaries) if chunk_summaries else ""

    # Vector store (optional RAG usage)
    vector_store, _ = initialize_vector_store()
    add_documents_to_store(vector_store, documents)

    return consolidated_summary, vector_store

# -------------------------------------------------------------------
# 4. Flexible Numeric Extraction (NER + Regex)
# -------------------------------------------------------------------
def load_spacy_model():
    print("Loading spaCy model for NER...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("SpaCy model 'en_core_web_lg' not found. Install via: `python -m spacy download en_core_web_lg`.")
        raise
    print("spaCy model loaded.")
    return nlp

def extract_flexible_numerics(summary_text, nlp):
    """
    Extract numeric references from summary_text using:
      - Regex patterns for certain financial keywords
      - spaCy NER for MONEY entities
    Store them in a list for a more flexible result
    """
    doc = nlp(summary_text)

    extracted_numerics = []

    # 1) Regex patterns for certain financial metrics
    regex_patterns = {
        "Regex_Revenue": r"(?:revenue(?:s)?|total\s+revenue)\s?(?:of|was|=)?\s?\$?([\d,\.]+)",
        "Regex_OperatingIncome": r"(?:operating\s+income|income\s+from\s+operations)\s?(?:of|was|=)?\s?\$?([\d,\.]+)",
        "Regex_NetIncome": r"(?:net\s+income|net\s+earnings)\s?(?:of|was|=)?\s?\$?([\d,\.]+)",
        "Regex_CostOfRevenue": r"(?:cost\s+of\s+revenue(?:s)?|cost\s+of\s+sales)\s?(?:of|was|=)?\s?\$?([\d,\.]+)",
        "Regex_TotalAssets": r"(?:total\s+assets)\s?(?:of|were|=)?\s?\$?([\d,\.]+)",
        "Regex_TotalLiabilities": r"(?:total\s+liabilities)\s?(?:of|were|=)?\s?\$?([\d,\.]+)",
        "Regex_StockholdersEquity": r"(?:stockholders'?(\s+)?equity|shareholders'?(\s+)?equity)\s?(?:of|was|=)?\s?\$?([\d,\.]+)",
        "Regex_CF_Operating": r"(?:cash\s+flow\s+from\s+operating\s+activities)\s?(?:of|=)?\s?\$?([\d,\.]+)",
        "Regex_CF_Financing": r"(?:cash\s+flow\s+from\s+financing\s+activities)\s?(?:of|=)?\s?\$?([\d,\.]+)",
        "Regex_CF_Investing": r"(?:cash\s+flow\s+from\s+investing\s+activities)\s?(?:of|=)?\s?\$?([\d,\.]+)",
    }

    for label, pattern in regex_patterns.items():
        matches = re.findall(pattern, summary_text, flags=re.IGNORECASE)
        for m in matches:
            numeric_val = m.replace(",", "")
            extracted_numerics.append({
                "label": label,
                "value": numeric_val,
                "source": "regex"
            })

    # 2) spaCy NER - record all money-like entities
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            ent_text = ent.text.strip().replace(",", "")
            extracted_numerics.append({
                "label": "MONEY_NER",
                "value": ent_text,
                "source": "ner",
                "context": ent.sent.text.strip()  # optional context
            })

    return {"extracted_numerics": extracted_numerics}

# -------------------------------------------------------------------
# 4.5 Clean numeric entries
# -------------------------------------------------------------------
def clean_numeric_entries(extracted_dict):
    """
    Takes a dict with key 'extracted_numerics' (a list of label/value dicts).
    - Remove empty or '.' values
    - Extract first numeric substring if multiple
    - Deduplicate (label, numeric_str)
    - Convert to float then back to string (strip .0 if integer)
    """
    numerics = extracted_dict.get("extracted_numerics", [])
    cleaned = []
    seen = set()  # track (label, numeric_str)

    for entry in numerics:
        raw_val = entry.get("value", "").strip()
        label = entry.get("label", "").strip()
        source = entry.get("source", "")
        context = entry.get("context", "")

        # Ignore trivial placeholders
        if not raw_val or raw_val in {".", "$", "$ ", " "}:
            continue

        # Find the *first* numeric substring
        match = re.search(r"[\d\.]+", raw_val)
        if not match:
            continue  # skip if no numeric substring

        numeric_str = match.group(0)  # e.g. '24667.00'
        # Attempt float parse
        try:
            fval = float(numeric_str)
        except ValueError:
            continue  # skip if parse fails

        # Convert back to string, removing trailing .0 if integral
        if fval.is_integer():
            numeric_str = str(int(fval))
        else:
            numeric_str = str(fval)

        # Deduplicate
        key = (label, numeric_str)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({
            "label": label,
            "value": numeric_str,
            "source": source,
            "context": context
        })

    return {"extracted_numerics": cleaned}

# -------------------------------------------------------------------
# 5. Save Output
# -------------------------------------------------------------------
def save_output_flexible(ticker, consolidated_summary, extracted_dict):
    summary_file = f"{ticker}_10q_summary.json"
    numerics_file = f"{ticker}_10q_flex_numerics.json"

    with open(summary_file, "w") as f:
        json.dump({"ConsolidatedSummary": consolidated_summary}, f, indent=4)
    print(f"Summary saved to {summary_file}")

    with open(numerics_file, "w") as f:
        json.dump(extracted_dict, f, indent=4)
    print(f"Extracted numerics saved to {numerics_file}")

# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    ticker = input("Enter the stock ticker symbol (e.g., MSFT): ").strip().upper()

    # 1) Initialize summarizer
    summarizer = initialize_summarizer_model()

    # 2) Summarize 10-Q
    consolidated_summary, vector_store = process_10q(ticker, summarizer)

    # 3) Load spaCy & do flexible numeric extraction
    nlp = load_spacy_model()
    extracted_dict = extract_flexible_numerics(consolidated_summary, nlp)

    # 4) Clean the extracted numerics
    cleaned_dict = clean_numeric_entries(extracted_dict)

    # 5) Save
    save_output_flexible(ticker, consolidated_summary, cleaned_dict)

if __name__ == "__main__":
    main()

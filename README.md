# **10-Q Analyzer: Extract Financial Insights with RAG and NER**

## **Overview**

The **10-Q Analyzer** is an AI-powered tool designed to analyze SEC 10-Q filings for publicly traded companies. It leverages **Retrieval-Augmented Generation (RAG)** and **Named Entity Recognition (NER)** to **summarize** lengthy financial reports and extract **key financial metrics** into a structured **JSON output**. 

This tool empowers retail investors by converting complex financial documents into actionable insights, enabling them to make more informed decisions for trading stocks or options.

---

## **Key Features**

1. **Input a Stock Ticker**: Provide the stock ticker symbol (e.g., MSFT for Microsoft) to analyze its latest 10-Q report.
2. **Summarize Long Documents**: Use the Flan-T5 summarization model to produce a concise summary of the report.
3. **Extract Key Financial Metrics**:
    - Utilize **SpaCy's NER** to detect monetary values.
    - Combine with **Regex Matching** for precision extraction of key financial metrics.
4. **Output**:
    - **Summarized Financial Insights**.
    - **Structured JSON** with extracted numeric data, labels, and context.

---

## **How It Works**

1. **Document Loading**:
   - Load the SEC 10-Q PDF document for the given stock ticker.
2. **Text Splitting**:
   - Split the document into smaller, manageable chunks for summarization.
3. **Summarization**:
   - Use a **Flan-T5** model to summarize each chunk.
4. **Named Entity Recognition (NER) & Regex Extraction**:
   - Identify monetary entities (e.g., revenue, expenses, net income).
   - Use regex patterns to capture key financial metrics.
5. **Cleaning and Structuring**:
   - Deduplicate and clean numeric entities.
   - Present data in a structured JSON format.
6. **Output**:
   - Save both the **summarized report** and **structured numerics**.

---

## **Example Output**

### **Input**: `MSFT` (Microsoft Stock Ticker)

**Sample JSON Output**:

```json
{
    "label": "MONEY_NER",
    "value": "50313",
    "source": "ner",
    "context": "Product $ 15,272 $ 15,535 Service and other $ 50,313 $ 40,982 (Reporting) (In millions, except per share amounts)"
},
{
    "label": "MONEY_NER",
    "value": "7544",
    "source": "ner",
    "context": "Research and development $7,544 6,659 Research and marketing $5,717 5,187 Research and development revenues amounted to a total of 7,544 6,659..."
},
{
    "label": "MONEY_NER",
    "value": "523013",
    "source": "ner",
    "context": "Total assets $ 523,013 $ 512,163 Total assets $ 523,013 $ 512,163 Liabilities and stockholders\u2019 equity..."
}
```

---

## **Setup Instructions**

### **1. Prerequisites**

- Python 3.8 or later
- Install required libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### **2. Project Structure**

```plaintext
.
├── app.py                   # Main application code
├── 10q_docs/                # Folder containing 10-Q PDFs
├── .env                     # Environment variables (ignored)
├── MSFT_10q_summary.json    # Example summary output
├── MSFT_10q_flex_numerics.json # Extracted numeric JSON output
└── requirements.txt         # Dependencies
```

### **3. Run the Application**

1. Place the 10-Q PDF file in the `10q_docs/{ticker}/` folder.
2. Run the script:
   ```bash
   python app.py
   ```
3. Enter the stock ticker when prompted (e.g., `MSFT`).

### **4. Outputs**

- **`{ticker}_10q_summary.json`**: Consolidated summary of the 10-Q report.
- **`{ticker}_10q_flex_numerics.json`**: Structured JSON with extracted financial metrics.

---

## **Technical Highlights**

### **RAG (Retrieval-Augmented Generation)**
- **Document Retrieval**: ChromaDB stores document embeddings for retrieval.
- **Summarization**: Uses **Flan-T5** to generate summaries of text chunks.

### **NER and Regex Extraction**
- **NER (SpaCy)**:
    - Extracts monetary values like revenue, income, and costs.
- **Regex Patterns**:
    - Ensures targeted extraction for metrics like:
        - **Revenue**: `total revenue, product revenue`
        - **Net Income**: `net income, earnings`
        - **Operating Cash Flow**: `cash flow from operating activities`

### **JSON Structuring**
- Combines the NER results and regex-matched values.
- Cleans, deduplicates, and validates numeric entries.

---

## **Use Cases**

1. **For Retail Investors**:
   - Quickly analyze complex 10-Q reports for financial metrics.
   - Use structured outputs for decision-making in trading.

2. **For Analysts**:
   - Automate extraction of metrics across multiple filings.
   - Save time on manual summarization and data entry.

3. **For FinTech Applications**:
   - Integrate structured JSON outputs into dashboards and tools.
   - Enhance analytics and recommendations for users.

4. **Cybersecurity/Other Domains**:
   - Adapt the framework for processing **logs** or **event data**.
   - Use RAG and NER for structured extraction of critical information.

---

## **Challenges and Considerations**

1. **Document Variability**:
   - 10-Q filings have inconsistent structures, requiring regex and NER tuning.

2. **Chunking and Summarization**:
   - Large documents need optimized chunk sizes for accurate summarization.

3. **Precision of Extraction**:
   - Combining NER with regex improves accuracy but may require domain-specific adjustments.

4. **Scalability**:
   - The pipeline can be extended for batch processing of multiple reports.

---

## **Future Improvements**

- **Model Enhancements**: Integrate fine-tuned LLMs for financial text.
- **Interactive UI**: Build a frontend for easier input and output visualization.
- **Automation**: Automate downloading of 10-Q reports from SEC’s EDGAR database.

---

**Transform the way you analyze financial data with AI-powered tools!** 🚀

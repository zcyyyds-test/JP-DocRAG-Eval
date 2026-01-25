# JP-DocRAG-Eval

A lightweight, evaluation-ready RAG (Retrieval-Augmented Generation) system optimized for Japanese technical documents. 
It demonstrates a full pipeline including ingestion, hybrid retrieval (BM25 + Dense), cross-encoder reranking, and LLM-based generation.

## File Structure

- **`web_demo.py`**: The main Streamlit application. Provides a chat interface, retrieval arena for comparing methods, and an evaluation dashboard.
- **`src/jprag/`**: Core logic package.
    - **`retrieve.py`**: Implements BM25, FAISS (Dense), and Hybrid retrieval (RRF).
    - **`rerank.py`**: Implements the Japanese Cross-Encoder for high-precision reranking.
    - **`llm.py`**: Wrapper for the LLM provider (Google Gemini, OpenAI, etc.).
    - **`bm25_logic.py`**: Custom BM25 implementation using character 3-grams for robust Japanese support without morphological analysis.
    - **`normalize.py`**: Text normalization utilities (Unicode NFKC, etc.).
    - **`chunk.py`**: Handles document splitting (sliding window with overlap) and text cleaning.
    - **`eval_run.py`**: Script to run offline evaluations (Recall@K, MRR) against gold-standard QA pairs.
    - **`eval_metrics.py`**: Calculation of retrieval metrics.
- **`manage.sh`**: Helper shell script to control the service process.
- **`config.yaml`**: (Optional) Configuration file for model parameters and API keys.

## Docker Setup

You can run the entire system using Docker Compose.

### Prequisites
- Docker & Docker Compose
- An API Key (e.g., Google Gemini)

### Instructions

1.  **Set Environment Variables**
    Create a `.env` file or export your API key:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

2.  **Build and Start**
    ```bash
    docker-compose up --build -d
    ```
    This will compile the image and start the container in detached mode.

3.  **Access the Web Interface**
    Open your browser and navigate to:
    [http://localhost:8501](http://localhost:8501)

4.  **Stop the Service**
    ```bash
    docker-compose down
    ```

## Local Service Management (Non-Docker)

> [!NOTE]
> `manage.sh` is a helper script for **local non-Docker** development only. 
> When running with `docker-compose`, this script is **NOT** used. Docker runs the application directly.
> To stop the Docker service, use `docker-compose down`.

If you prefer to run locally (e.g., for development):

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Service**
    Use the helper script `manage.sh`:
    ```bash
    # Start in background
    ./manage.sh start
    
    # Check status
    ./manage.sh status
    
    # View logs
    ./manage.sh logs
    
    # Stop
    ./manage.sh stop
    ```
    Alternatively, run directly with Streamlit:
    ```bash
    streamlit run web_demo.py
    ```

## Notes

- **Reranking**: The system includes an optional Reranker (Cross-Encoder). You can enable it in the "Chat" capability of the web interface via the **"Enable Cross-Encoder Rerank"** toggle. It significantly improves precision at the cost of slight latency.
- **Data Persistence**: The `data/` and `artifacts/` directories are mounted in Docker to persist your indices and logs.

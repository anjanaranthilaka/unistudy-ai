# UniStudy AI: Local University Document & Study Assistant

UniStudy AI is an offline, privacy-first academic study assistant that allows university students to interact with lecture notes, textbooks, and slide decks. Powered by a local Large Language Model via **Ollama** (`llama3.2:3b`), UniStudy AI performs document question answering, structured summarization, multiple-choice quiz generation, and student answer evaluation without relying on paid third-party APIs or cloud subscriptions.

---

## 1. System Architecture

The application adopts a modular, transparent pipeline without heavy wrapper frameworks (such as LangChain or CrewAI). Every agent step, tool call, and state transition is implemented in standard Python:


```

[Student Interface (Streamlit UI)]
│
▼
[StudyAgent (agents/study_agent.py)]
├─ 1. Route Intent (Structured JSON / Rule Fallback)
├─ 2. Tool Execution (tools/)
│     ├─ read_document (pypdf text & page extraction)
│     ├─ search_documents (In-memory term-frequency ranking)
│     ├─ summarize_document (Structured study synthesis)
│     ├─ generate_quiz (Constrained JSON MCQ generation)
│     └─ evaluate_answer (Formative grading against source notes)
├─ 3. Local LLM Generation (llm/ollama_client.py -> http://localhost:11434)
└─ 4. Observable Trace Recording
│
▼
[UI Feedback: Grounded Answer + Visible Agent Trace + Cited Passages]

```

---

## 2. Key Features

- **Document Processing:** Ingests `.pdf`, `.txt`, and `.md` files. Preserves page boundaries and alerts students if a PDF is scanned (image-only).
- **Lightweight Retrieval (RAG):** Splits text into configurable chunks (300 words with 50-word overlap) and ranks passages using an in-memory term frequency scoring function.
- **Evidence-Grounded QA:** Injects retrieved passages into safe `<evidence>` blocks with strict prompt rules that prevent hallucinations.
- **Structured Practice Quizzes:** Produces validated multiple-choice questions with answer keys and rationale.
- **Formative Answer Grading:** Compares student explanations to lecture reference texts, outputting constructive feedback, approximate scores, and model answers.
- **Visible Tool Trace:** Exposes the agent's internal decisions, tool names, parameters, and execution statuses directly in the UI.

---

## 3. Technology Stack

- **Language:** Python 3.10+ (Tested on Python 3.14)
- **Local LLM Engine:** Ollama (Running `llama3.2:3b`)
- **User Interface:** Streamlit
- **PDF Extraction:** `pypdf`
- **HTTP Client:** Python standard `requests`
- **Test Framework:** `pytest`

---

## 4. Setup and Installation

### Prerequisites
1. Install [Ollama](https://ollama.com/) for your operating system.
2. Confirm Ollama is running and pull the default model:
   ```bash
   ollama pull llama3.2:3b

```

### Project Setup

1. Clone or extract the repository:
```bash
cd unistudy-ai

```


2. Create and activate a Python virtual environment:
* **Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```


* **macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate

```




3. Install required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt

```



---

## 5. Execution

Launch the Streamlit web interface:

```bash
streamlit run app.py

```

Open `http://localhost:8501` in your browser.

---

## 6. Running the Automated Test Suite

UniStudy AI includes isolated unit tests with mock fixtures that run independently of whether Ollama is active:

```bash
pytest tests/ -v

```

---

## 7. Example Prompts & Workflow

1. **Document Question Answering:**
* *Prompt:* `"What happens if the learning rate in gradient descent is set too high?"`
* *Expected Flow:* Agent executes `router` -> `search_documents` -> retrieves chunk from Page 1 -> `ollama_generate` outputs citation `[lecture1_ml.txt - Page 1]`.


2. **Practice Quiz Generation:**
* *Prompt:* `"Generate a 3-question multiple-choice quiz on supervised learning and overfitting."`
* *Expected Flow:* Agent executes `router` -> `search_documents` -> `generate_quiz` -> renders formatted MCQs with options and explanations.


3. **Answer Evaluation:**
* *Prompt:* `"Evaluate my answer: Overfitting happens when a model fits training data too closely, including noise, failing to generalize to new data."`
* *Expected Flow:* Agent executes `router` -> `search_documents` -> `evaluate_answer` -> returns formative score, strengths, and sample model answer.



---

## 8. Known Limitations

* **Scanned Documents:** Slide decks stored as bitmap images without a digital text layer require external OCR (such as Tesseract), which is not bundled.
* **Keyword Retrieval vs. Dense Embeddings:** Basic term frequency matching relies on word overlap; synonyms or paraphrased queries without shared tokens may rank lower than dense vector embeddings.

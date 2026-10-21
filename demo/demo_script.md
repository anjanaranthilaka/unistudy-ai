# UniStudy AI: 3–5 Minute Video Demonstration Script

## Video Guidelines
- **Target Duration:** 3 minutes 30 seconds to 4 minutes 30 seconds.
- **Resolution:** 1080p (Full Screen).
- **Prerequisites:** Ensure Ollama is running (`ollama list` shows `llama3.2:3b`), and launch the app in PowerShell: `streamlit run app.py`.

---

### Segment 1: Introduction & System Architecture (0:00 - 0:45)
- **Visual:** Display the browser showing the UniStudy AI interface at `http://localhost:8501`.
- **Narration:**
  > "Hello. Today I am demonstrating UniStudy AI, a local, privacy-preserving university study assistant. 
  > The application runs completely offline using Ollama and the llama3.2:3b model on consumer hardware. 
  > In the left sidebar, we can see the system health indicator confirming that Ollama is online and running locally on port 11434. 
  > Unlike cloud systems, no lecture notes or student queries are ever sent to third-party APIs. 
  > Our architecture uses clean, transparent Python: an in-memory chunking and retrieval engine, an allowlisted tool registry, and an observable agent coordinator."

---

### Segment 2: Document Ingestion & Task 1: Grounded QA (0:45 - 2:00)
- **Visual:** 
  1. Drag and drop `sample_documents/lecture1_ml.txt` into the file uploader.
  2. Point out the success banner: `1 document(s) indexed` showing total pages and chunks.
  3. Type into the chat input:
     `"What happens if the learning rate in gradient descent is too large?"`
  4. Press Enter.
- **Narration:**
  > "First, I will upload our sample lecture notes on Machine Learning. The document reader extracts the text and creates overlapping 300-word chunks while preserving page metadata.
  > Now, I will ask a specific question: 'What happens if the learning rate in gradient descent is too large?'
  > As the model responds, notice the answer is grounded directly in the uploaded text: it explains that the algorithm may overshoot the minimum and diverge.
  > Below the answer, we expand the 'Cited Source Passages' to see the exact file and page number retrieved.
  > Most importantly, let's open the 'Agent Execution Trace'. Here we see every internal step:
  > Step 1: The router identified this as an 'ask_question' intent.
  > Step 2: The keyword search tool located the matching passage.
  > Step 3: Ollama synthesized the grounded answer.
  > Everything is visible and verifiable."

---

### Segment 3: Task 2: Multi-Step Active Revision Task (2:00 - 3:15)
- **Visual:**
  1. Type or click the button to submit:
     `"Generate a 3-question multiple choice practice quiz from my notes."`
  2. Let the quiz render on screen.
  3. Immediately follow up in the chat input:
     `"Evaluate my answer: Overfitting is when a model learns training noise too closely and fails to predict new data."`
  4. Press Enter.
- **Narration:**
  > "Now we demonstrate a multi-step revision workflow.
  > First, I ask the assistant to generate a 3-question multiple choice practice quiz.
  > The agent routes this request to the quiz generator tool, which prompts Ollama in constrained JSON mode. The system renders three multiple-choice questions with answer keys and explanations grounded in the notes.
  > Next, I will test my own knowledge by submitting an answer for evaluation.
  > The agent executes our answer evaluation tool, comparing my response to the reference text.
  > It returns an estimated score, highlights my strengths, identifies missing concepts, and provides a sample model answer. It also reminds the student that this feedback is for study guidance."

---

### Segment 4: Automated Testing & Conclusion (3:15 - 3:50)
- **Visual:** Switch briefly to the PowerShell terminal. Run `pytest tests/ -v`. Show all 15 tests passing in green.
- **Narration:**
  > "Behind the interface, the system is backed by comprehensive unit tests covering document parsing, chunking, retrieval scoring, and mock agent loops. All 15 tests pass without requiring an external database.
  > In conclusion, UniStudy AI provides students with an offline, transparent study assistant that turns lecture notes into active revision tools while keeping all data private. Thank you."

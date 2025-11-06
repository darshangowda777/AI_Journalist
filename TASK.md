# AI Journalist 3.5 

## Objective
You are building an *AI Journalist* — an intelligent agentic system that reads, reasons, and verifies information from large news datasets.

Your system must:
1. Analyze text using **classical NLP** (POS, NER, TF-IDF, sentiment).  
2. Retrieve evidence via **Hybrid RAG** (BM25 + Chroma).  
3. Generate **fact-grounded summaries** with citations.  
4. Verify claims using **semantic similarity + NLI**.  
5. Orchestrate all components via a **LangGraph pipeline**.  
6. Log your reasoning process and confidence.

---

##  Dataset
Use the provided dataset:
data/news_corpus.csv # 200,000 news articles
data/queries.json # query prompts for your agent


Each record has:
- `ID`: article ID  
- `title`: short news headline  
- `source`: publisher name  
- `category`: b = business, t = science/tech, e = entertainment, m = health  
- `date`: publication date  

---

##  Repository Structure
You’ll work inside this scaffold:

src/
│ classical_agent.py → implement classical NLP pipeline
│ retriever_hybrid.py → build BM25 + Chroma hybrid retriever
│ summarizer_agent.py → generate factual summaries with citations
│ verifier_agent.py → verify factual precision, contradiction, consistency
│ orchestrator.py → connect all agents with LangGraph
│ utils.py → optional helper functions
data/
│ news_corpus.csv
│ queries.json
results/
│ metrics.json, trace.json, plots/


---

## Tasks

### **Task 1 — Classical NLP Agent**
Implement tokenization, POS tagging, NER, TF-IDF keyword extraction, and sentiment scoring.
- Input: sample texts from `news_corpus.csv`
- Output: `results/classical_output.json`

### **Task 2 — Hybrid Retriever**
Implement retrieval using:
- Sparse: `rank_bm25`
- Dense: `Chroma` + `sentence-transformers/all-MiniLM-L6-v2`
- Fusion formula:
  \[
  score_{hybrid} = \alpha \cdot dense + (1 - \alpha) \cdot sparse
  \]
- Output: JSON list of top-k documents with scores.

### **Task 3 — Summarizer / QA Agent**
- Use any instruction-tuned model (Flan-T5, Mistral-7B, etc.).
- Prompt must output claims + citations.
- Example:

- Output: `results/generated_summary.txt`

### **Task 4 — Verifier Agent**
- Compare generated summary with source docs.
- Compute:
- Factual precision (semantic similarity > 0.8)
- Contradiction rate (NLI)
- Temporal consistency
- Output: `results/metrics.json`

### **Task 5 — Orchestration**
- Use `LangGraph` to connect all agents.
- Define flow:
Classical NLP → Retrieve → Summarize → Verify
↳ Retry if confidence < 0.7

- Log entire trace: `results/trace.json`

### **Task 6 — Visualization & Presentation**
- Visualize:
- BM25 vs Chroma score weights
- LangGraph pipeline trace


---

## Evaluation Rubric (100 pts)
| Component | Weight |
|------------|---------|
| Classical NLP | 15 |
| Hybrid Retrieval | 15 |
| Multi-Hop Reasoning | 10 |
| Summarization & Citations | 20 |
| Fact Verification | 15 |
| Orchestration & Adaptivity | 15 |
| Visualization | 5 |
| Presentation | 5 |

Bonus +5 → Interactive Gradio demo or novel explainability feature.

---

## Deliverables
- `src/` folder with working code
- `results/` folder with outputs (`trace.json`, `metrics.json`, etc.)
- Short README (team name, approach, instructions)
- GitHub repo link or ZIP submission

---

## Optional Extensions
- Add named-entity-driven **multi-hop retrieval**
- Implement **confidence-based retry**
- Visualize **retrieval graph**
- Run **cross-category analysis** using your own query from `queries.json`

---

## Duration
**3 – 4 hours**  
Teams of 2–3 students.

---

## Submission
Upload your completed repository (or GitHub link) to Moodle / SRH GitHub Classroom.  
Name your repo:

# Algorithm Design Project: Prompt Optimization (Group 11)

**Repository:** `Algorithm-Project-PromptOptimization-Group11`

This project treats **prompt optimization** as a classic **search and optimization problem** over a *discrete prompt space*. We generate candidate prompts using explicit edit operators, evaluate them with well-defined metrics on QA and summarization datasets, and optimize them using classic search and optimization algorithms such as **Hill Climbing**, **BFS**, and **Beam Search**, enhanced with heuristic constraints.

---

## 1. Problem Definition

### Input
- **Task**: `T ∈ {qa, summarization}`
- **Dataset**: `D = {(x_i, y_i)}`
- **Initial Prompt**: `p0`
- **Prompt Generator**: `G` (edit operators that generate neighboring prompts)
- **Evaluator**: `E` (LLM backend + evaluation metric)
- **Budget**: `B` (maximum number of prompt evaluations)

### Output
- **Best Prompt**: `p*`
- **Best Score**: `m(p*)`
- **Trace Logs**: records of evaluated prompts (used for convergence plots)

---

## 2. Algorithms Implemented (Phase 1)

### Search Algorithms
- **Breadth-First Search (BFS)** on prompt space  
  `src/algorithms/bfs.py`
- **Beam Search**  
  `src/algorithms/beam_search.py`
- **Heuristic-Constrained Search** (length + novelty pruning)  
  `src/algorithms/heuristics.py`

### Optimization Algorithm
- **Hill Climbing**  
  `src/algorithms/hill_climbing.py`

---

## 3. Evaluation Metrics

### Question Answering (QA)
- **Exact Match (EM)** after normalization (case, whitespace, punctuation)

### Summarization
- **ROUGE-L (F1)** based on token-level Longest Common Subsequence (LCS)

---

## 4. LLM Backends (Ollama + MockLLM)

- **OllamaLLM** (`src/evaluation/ollama_llm.py`)  
  Local, cost-free LLM used for Phase 1 and Phase 2 experiments.

- **MockLLM** (`src/evaluation/mock_llm.py`)  
  Deterministic backend for unit tests (fast and reproducible).

> **Note:** Unit tests rely on `MockLLM` to avoid nondeterminism. Real experiments can be run with `OllamaLLM`.

---

## 5. Repository Structure

```

src/
├── algorithms/
├── evaluation/
├── prompt/
├── utils/
├── tests/
scripts/
reports/
results/

````

- Generated artifacts such as `results/trace_*.csv` are ignored via `.gitignore`.
- Heavy datasets or large outputs **must not** be committed to the repository.

---

## 6. Setup Instructions

### Requirements

Install Python dependencies:
```bash
pip install -r requirements.txt
````

Optional (for real LLM runs):

* Install **Ollama**
* Ensure Ollama is running
* Set the model:

```bash
export OLLAMA_MODEL="llama3.1:8b"
```

---

## 7. How to Run

### Run Unit Tests

```bash
python -m pytest -q
```

### Run Hill Climbing

```bash
python -m scripts.run_hill_climb --task qa --backend mock --budget 60
python -m scripts.run_hill_climb --task summarization --backend mock --budget 60
```

### Unified Search Runner (BFS / Beam / Hill)

```bash
python -m scripts.run_search --task qa --backend mock --algo hill --budget 80

python -m scripts.run_search --task qa --backend mock --algo bfs --budget 120 --max_depth 4

python -m scripts.run_search --task qa --backend mock --algo beam --budget 120 --max_depth 4 --beam_width 5
```

### Heuristic-Constrained Search

```bash
python -m scripts.run_search --task qa --backend mock --algo bfs --budget 120 --max_depth 4 --heuristics --novelty 0.90

python -m scripts.run_search --task qa --backend mock --algo beam --budget 120 --max_depth 4 --beam_width 5 --heuristics --novelty 0.90
```

### Using the Ollama Backend

```bash
python -m scripts.run_search --task qa --backend ollama --algo beam --budget 60 --max_depth 3 --beam_width 3
```

---

## 8. Sample Inputs and Outputs

* **Input datasets**: `src/utils/datasets.py`
* **Outputs**:

  * Best prompt printed to the terminal
  * Trace file saved to `results/trace_<run_id>.csv` (ignored by Git)

---

## 9. GitHub Workflow (Course Rules)

* Branch per phase:

  * `phase-1` → PR → `main`
  * `phase-2` → PR → `main`
* Each phase must include **at least 5 meaningful commits**

  * Avoid vague messages such as `final`, `fix`, or `done`
* Do **not** commit large artifacts

  * Store heavy data externally (e.g., Google Drive, HuggingFace)

---

## 10. Reports

* **Phase 1 Report**: `reports/phase1.tex`
* Build instructions available in `reports/README.md`



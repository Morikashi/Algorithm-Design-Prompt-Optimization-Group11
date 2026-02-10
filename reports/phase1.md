# Reports & Artifacts

This directory contains the **Phase 1 report** and supporting instructions for generating the final PDF, along with notes on experimental artifacts and reproducibility.

---

## 1. Phase 1 Report (LaTeX)

### Main File
- `phase1.tex`

### Build Instructions
From the project root:
```bash
cd reports
pdflatex phase1.tex
````

### Output

* `phase1.pdf`

> You may need to run `pdflatex` twice to resolve references, depending on your LaTeX setup.

---

## 2. Results Artifacts

Convergence traces are generated when running the search and optimization scripts.

### Location

```
results/trace_<run_id>.csv
```

### Notes

* These trace files record prompt evaluations over time.
* They are **intentionally ignored by Git** (see the root `.gitignore`) to keep the repository clean and lightweight.
* Do **not** commit generated trace files or large experimental outputs.

---

## 3. Reproducing Phase 1 Runs

### Recommended (run scripts as modules)

```bash
python -m scripts.run_hill_climb --task qa --backend mock --budget 60

python -m scripts.run_search --task qa --backend mock --algo beam --budget 120 --max_depth 4 --beam_width 5
```

### Using the Ollama Backend

Set the model first:

```bash
export OLLAMA_MODEL="llama3.1:8b"
```

Run an example experiment:

```bash
python -m scripts.run_search --task summarization --backend ollama --algo bfs --budget 60 --max_depth 3
```

---

## 4. Phase 2 Preview (Not Implemented Yet)

Phase 2 will extend the project with:

* Larger real-world test sets (≥ 10 inputs per task)
* Runtime and memory profiling
* Convergence plots and performance tables
* Optional algorithms:

  * Simulated Annealing
  * Genetic Algorithms
* Final report including:

  * System diagram
  * Empirical analysis and comparisons

---

## 5. Quick “Step 6 Done” Checklist

After adding or updating these files, make sure to:

1. **Verify `.gitignore`**

   * Ensure it ignores experimental artifacts:

     ```
     results/trace_*.csv
     ```
   * Optionally ignore LaTeX build files:

     ```
     reports/*.aux
     reports/*.log
     reports/*.out
     reports/*.pdf
     ```

2. **Build the report**

   ```bash
   cd reports
   pdflatex phase1.tex
   ```

3. **Commit with a meaningful message**, for example:

   ```
   write phase 1 report in latex and update repository readmes
   ```

4. **Push changes and open a Pull Request**

   * From `phase-1` → `main`
   * Clearly describe:

     * What Phase 1 completed
     * What remains for Phase 2



# 🚀 Intelligent Candidate Discovery & Ranking Engine
**Official Submission for the Redrob AI Hackathon (v4)**

> A highly optimized, lightweight, and hybrid Candidate Ranking Engine designed to process 100,000+ candidates and output the top 100 best fits within strict compute constraints.

Unlike heavy LLM-per-candidate pipelines that fail production constraints, this engine is aggressively optimized to run on a **CPU-only environment within a strict 5-minute limit and 16GB RAM**, simulating a real-world, highly scalable HR-tech architecture.

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `rank.py` | The core Python ranking pipeline and scoring engine. |
| `requirements.txt` | Minimal Python dependencies (`pandas`, `scikit-learn`). |
| `submission_metadata.yaml` | Team details, AI declarations, and compute environment specs. |
| `README.md` | Project documentation and execution instructions. |

---

## 🧠 System Architecture & Methodology

The pipeline follows a strict **5-stage funnel approach** to prune bad fits early and surface top talent efficiently:

1. **🚫 Stage 1: The Trap Filters (Zero-Shot Disqualification)**
   * **Honeypot Filter:** Identifies logically impossible profiles (e.g., skill durations exceeding a candidate's total years of experience) to stay well under the >10% honeypot trap limit.
   * **Ghost Filter:** Drops candidates who have been inactive for over 6 months, ensuring we only rank highly available talent.

2. **⚖️ Stage 2: Smart Feature Engineering & Penalties**
   * **Title Heuristics:** Automatically detects and heavily penalizes irrelevant job titles (e.g., Marketing, HR, Sales) even if their resumes are stuffed with AI keywords.
   * **Product vs. Service Calibration:** Applies a slight penalty to profiles built entirely in pure-services consulting firms, aligning with the JD's preference.

3. **⚙️ Stage 3: CPU-Optimized Text Vectorization**
   * Uses **TF-IDF with N-grams (1, 2)** and `cosine_similarity` to perform semantic matching in seconds without relying on external API calls or GPUs.
   * Engineered to safely handle JSON structural variations (bulletproof parsing for skills as both dicts and flat strings).

4. **🧮 Stage 4: Hybrid Scoring Mathematics**
   * Final Score = `(Semantic Technical Match * 0.70) + (Behavioral Activity * 0.30)`
   * **Behavioral Signals:** Rewards active developers based on Recruiter Response Rate and Normalized GitHub Activity.

5. **📝 Stage 5: Explainability Generation**
   * Generates deterministic, variable-driven reasoning strings for the Top 100 candidates to satisfy manual review constraints (explaining *why* they ranked high based on exact metrics).

---

## 💻 Technical Constraints Satisfied
- ✅ **Compute:** CPU-only (No GPU required).
- ✅ **Time:** Executes the entire 100k candidate pipeline in < 2 minutes (Limit: 5 mins).
- ✅ **Memory:** Operates comfortably within 16 GB RAM.
- ✅ **Network:** 100% offline ranking execution (Zero external API dependencies).

---

## 🛠️ How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution
Place your candidates.jsonl (or .gz) file and job_description.md in the root directory, then run:

```bash
python rank.py
```
The script will automatically generate a CSV file containing the uniquely ranked top 100 candidates along with their generated reasoning.

### 3. Validation
To verify the output matches the exact Hackathon specifications:

```bash
python validate_submission.py <your_output_filename>.csv
```
🌐 Live Sandbox Demo
A working hosted environment (Hugging Face Space) has been created to test this ranking engine on a small candidate sample.

Sandbox Link: [https://huggingface.co/spaces/devwithsk/redrob-ranker]
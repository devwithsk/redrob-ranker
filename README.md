# Intelligent Candidate Discovery & Ranking Engine 🚀
**Built for the Redrob AI Hackathon (v4)**

This repository contains a highly optimized, lightweight, and hybrid Candidate Ranking Engine designed to process a pool of 100,000+ candidates and output the top 100 best fits based on a specific Job Description (JD). 

Unlike heavy LLM-per-candidate pipelines that fail production constraints, this engine is aggressively optimized to run on a **CPU-only environment within the strict 5-minute and 16GB RAM constraints**, simulating a real-world, highly scalable HR-tech architecture.

---

## 🧠 System Architecture & Methodology

The pipeline follows a strict 5-stage funnel approach to prune bad fits early and surface top talent:

### Stage 1: Ingestion & The Trap Filters (Zero-Shot Disqualification)
Before applying any compute-heavy algorithms, the system purges irrelevant data:
* **The Honeypot Filter:** Identifies logically impossible profiles (e.g., skill durations exceeding a candidate's total years of experience) to avoid the >10% honeypot trap limit.
* **The Ghost Filter:** Drops candidates who have been inactive for over 6 months, ensuring we only rank highly available talent.

### Stage 2: Smart Feature Engineering & Penalty Engine
* **Title Heuristics:** Automatically detects and heavily penalizes irrelevant job titles (e.g., Marketing, HR, Sales) even if they stuffed their resume with AI keywords.
* **Product vs. Service Calibration:** Applies a slight penalty to profiles built entirely in purely service-based consulting firms, aligning with the JD's preference for product-company experience.

### Stage 3: CPU-Optimized Text Vectorization (The Core)
* Extracts core JD parameters and compares them against a concatenated string of the candidate's summary, career history, and skills.
* Uses **TF-IDF with N-grams (1, 2)** and `cosine_similarity` to perform semantic matching in seconds without relying on external API calls or GPUs.
* *Bulletproof Parsing:* Engineered to safely handle JSON structural variations (handling skills as both lists of dicts and flat strings).

### Stage 4: Hybrid Scoring Mathematics
The final composite score is calculated using a weighted formula:
`Final Score = (Semantic Technical Match * 0.70) + (Behavioral Activity * 0.30)`
* **Behavioral Signals:** Includes Recruiter Response Rate and Normalized GitHub Activity to reward actively engaged developers.

### Stage 5: Explainability Generation
To satisfy manual review constraints, the system generates deterministic, variable-driven reasoning strings for the Top 100 candidates, clearly explaining exactly *why* they ranked high (e.g., exact years of experience, response rate, and core role validation).

---

## 💻 Technical Constraints Satisfied
- [x] **Compute:** CPU-only (No GPU required).
- [x] **Time:** Executes the entire 100k candidate pipeline well under the 5-minute limit.
- [x] **Memory:** Operates comfortably within 16 GB RAM.
- [x] **Network:** 100% offline ranking execution (Zero external API dependencies during the ranking phase).

---

## 🛠️ How to Run Locally

### Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```
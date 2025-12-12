# PedagoReLearn


**© 2025 Thomas F. Hallmark**  
Licensed under the [MIT License](LICENSE)

---

## 🌍 Overview

**PedagoReLearn** is an **AI-driven reinforcement learning (RL) framework** for adaptive cross-cultural tutoring.  
It models cultural competence training as a *state–action–reward* process where an agent learns when to **teach**, **review**, or **quiz** learners across domains such as etiquette, privacy, work, and travel.

Grounded in **John Dewey’s progressive education theory**, the system demonstrates how pedagogical strategies can *emerge autonomously through experience*, advancing long-term mastery and retention.

---

## 🎓 Project Summary

PedagoReLearn formulates tutoring as a **Markov Decision Process (MDP)**:

| Component | Description |
|------------|-------------|
| **States** | Learner mastery levels and recency of review for each cultural rule |
| **Actions** | `teach`, `quiz`, `review`, or `no-op` (pause) |
| **Rewards** | Reflect learning success, retention, and teaching efficiency |

Each YAML file under `/rules/` defines a **cultural knowledge domain** (e.g., workplace etiquette, travel behavior, hygiene norms).  
The **Gymnasium environment** (`pedagorelearn_env.py`) interprets these as interactive learning topics.

---

## 📁 Repository Structure
```
PedagoReLearn/
│
├── agents/                         # RL agent implementations
│   ├── sarsa_agent.py              # SARSA(0) on-policy learner
│   └── random_agent.py             # Random baseline policy
│
├── archive/                        # Backup versions & logs
│
├── docs/                           # Proposals, outlines, and references
│   ├── Proposal - PedagoReLearn.pdf
│   ├── Outline R5 - Project Proposal (with Dewey).docx
│   └── README_German_Cultural_Rules_Handbook_Final.pdf
│
├── experiments/                    # Experimental scripts
│   ├── compare_sarsa_versions.py
│   ├── analyze_csv.py
│   └── analyze_runs.py
│
├── plots/                          # Generated performance figures
│
├── results/                        # CSV output files by seed/run
│
├── rules/                          # YAML cultural knowledge base
│   ├── work_professional.yaml
│   ├── transport_travel.yaml
│   ├── digital_privacy.yaml
│   ├── religion_customs.yaml
│   ├── economy_society.yaml
│   ├── hygiene.yaml
│   ├── emergency_legal.yaml
│   └── …
│
├── trace_results/                  # Training logs and evaluation data
│   ├── cultural_rule_validator.py
│   └── …
│
├── LICENSE
├── README.md
├── requirements.txt
│
├── pedagorelearn_env.py            # Core Gymnasium environment
├── rules_loader.py                 # YAML rule loader and validator
│
├── student_model_sarsa.py          # Simulated student model
├── student_model_complete.py       # Extended learner model (alt.)
│
├── train_runner.py                 # Training control script
├── train_eval.py                   # Evaluation & comparison logic
│
├── tutor_train_sarsa_rewarded.py   # Main SARSA training driver
├── tutor_baselines.py              # Baseline policies
├── run_training.sh                 # Shell automation script
└── plot_trace_results.py           # Visualization utilities
```
---

## ⚙️ Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train SARSA agent
python tutor_train_sarsa_rewarded.py

# 4. Analyze or visualize results
python plot_trace_results.py
python compare_sarsa_versions.py

---

Expected Output Example

Episode 500 | Avg reward = 118.6 | Steps-to-mastery = 47 | Accuracy = 0.91
Aggregated policy surpasses fixed curriculum baseline.

🧠 Tech Stack
	•	Python 3.10+
	•	Gymnasium ≥ 0.29
	•	NumPy ≥ 1.23
	•	Matplotlib ≥ 3.7
	•	PyYAML for rule parsing and validation

⸻

🚀 Current Progress
	•	✅ Full Gymnasium-compliant environment with stochastic student model
	•	✅ Reward shaping, mastery tracking, and forgetting dynamics verified
	•	✅ Functional SARSA(0) agent with ε-greedy exploration
	•	✅ Aggregation and baseline comparison scripts tested
	•	✅ Week-by-week reproducible results stored under /results/

Next steps: aggregation ablations, statistical analysis, and documentation polish for final submission.

⸻

🔬 Future Directions
	•	Full aggregation ablation across four schemes
	•	Sensitivity analysis for α, γ, and ε-decay parameters
	•	Integration of heuristic spaced-repetition baseline
	•	Policy interpretability visualization (heatmaps, frequency plots)
	•	Expansion to additional cultural domains and learner models
	
	🙏 Acknowledgments

Developed for CSCE 642: Reinforcement Learning (Fall 2025)
**Texas A&M University**

Conceptual design and implementation by **Thomas F. Hallmark** and **Jun Kwon**.
>**AUTHOR BIOGRAPHIES**
>
> **Hallmark, T. F. (2025).** | thomas.hallmark@tamu.edu
>
>Thomas F. Hallmark is a doctoral student in Curriculum and Instruction with a cognate in Engineering Education in the Department of Teaching, Learning, and Culture at Texas A&M University. He holds degrees in Legal Studies and Business Administration (MBA) and brings more than 30 years of experience in the nuclear and utilities industries. His research focuses on the integration of artificial intelligence and reinforcement learning in engineering and STEM education, emphasizing adaptive tutoring systems, veteran transitions, and cross-cultural learning. Hallmark’s work combines pedagogical theory with computational modeling to design human-centered AI learning environments.
>
> **Kwon, J. (2025).**
>
>Jun Kwon is a graduate student in Computer Science and Engineering at Texas A&M University, specializing in machine learning and artificial intelligence applications for education and human-computer interaction. His research interests include reinforcement learning algorithms, neural network optimization, and adaptive feedback mechanisms in educational software. Kwon contributes to the computational architecture and algorithmic implementation of PedagoReLearn, focusing on model design, environment development, and performance evaluation across multiple RL frameworks.
> 
>> **Joint Contribution**
>Hallmark and Kwon collaboratively developed the conceptual framework and technical implementation of PedagoReLearn, merging educational theory and AI engineering to advance research in adaptive tutoring systems and cultural-learning reinforcement models.

⸻

📜 GitHub Description

Adaptive RL tutoring system modeling cultural learning through Dewey-inspired state, action, and reward design.

⸻

🤖 AI Use Disclaimer

Artificial intelligence (AI) tools—including ChatGPT—were used only for grammar, formatting, and document organization.
All intellectual content (code, methodology, analysis) is the original work of the authors and complies with Texas A&M University academic integrity standards.



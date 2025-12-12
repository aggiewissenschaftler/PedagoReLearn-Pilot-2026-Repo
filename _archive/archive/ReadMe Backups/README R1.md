# PedagoReLearn

**Copyright © 2025 Thomas F. Hallmark**  
Licensed under the MIT License (see [LICENSE](LICENSE)).

## Overview

PedagoReLearn is an AI-driven reinforcement learning framework for adaptive cross-cultural tutoring. It models cultural competence training as a state–action–reward process, where an agent learns when to teach, review, or quiz learners across domains such as etiquette, privacy, work, and travel. Grounded in Dewey’s progressive education theory, the system explores how effective pedagogical strategies can emerge autonomously through experience to promote long-term mastery and retention.

---

## Project Overview

PedagoReLearn models tutoring as a **Markov Decision Process (MDP)**:

- **States:** learner mastery levels and recency of review for each cultural rule  
- **Actions:** teach, quiz, review, or pause (no-op)  
- **Rewards:** reflect learning success, retention, and teaching efficiency  

Each YAML file under `/rules/` defines a domain of required cultural behaviors (e.g., workplace, travel, or hygiene norms).  
The RL environment (`pedagorelearn_env.py`) interprets these as learning topics.

---

Current Stage (Week 8–9)
•	Environment fully Gymnasium-compliant with stochastic student model
•	Reward shaping, mastery tracking, and forgetting dynamics validated
•	Tabular SARSA(0) agent functional with ε-greedy exploration
•	State aggregation and baseline comparison scripts under evaluation
•	Week-by-week results archived with reproducible seeds and CSV logs

Upcoming work includes completing aggregation ablations, full statistical analysis, and documentation polish for submission.

---

## 📁 Repository Structure

```
PedagoReLearn/
│
├── agents/                        										# RL agents
│   ├── sarsa_agent.py              									# SARSA(0) on-policy learner
│   ├── random_agent.py             								# Baseline random policy
│
├── archive/                        										# Backups & previous versions
│   ├── pedagorelearn_env_rewarded.py
│   └── Backup Code / Logs
│
├── docs/                           										# Reports, proposals, and handbook
│   ├── Proposal - PedagoReLearn.pdf
│   ├── Outline R5 - Project Proposal (with Dewey).docx
│   ├── PedagoReLearn Project Management.docx
│   ├── Week 7 Achieved.docx
│   └── README_German_Cultural_Rules_Handbook_Final.pdf
│
├── experiments/                    									# Comparison & analysis scripts
│   ├── compare_sarsa_aggregation.py
│   ├── analysis_week.py
│
├── plots/                          										# Generated visualizations
│   ├── week9_curve_acc.png
│   ├── week9_bar_steps_mean.png
│   └── ...
│
├── results/                        										# CSV outputs by seed and scheme
│   ├── curves_full_seed*.csv
│   ├── curves_aggregated_seed*.csv
│   └── aggregation_comparison.csv
│
├── rules/                          										# YAML cultural knowledge base
│   ├── work_professional.yaml
│   ├── transport_travel.yaml
│   ├── digital_privacy.yaml
│   ├── religion_customs.yaml
│   ├── economy_society.yaml
│   ├── hygiene.yaml
│   ├── emergency_legal.yaml
│   └── ... (20+ rule sets total)
│
├── trace_results/                  									# Training, evaluation, and logs
│   ├── logs/
│   │   └── sarsa_rewarded_train_log_*.csv
│   ├── analyze_csv.py
│   ├── compare_sarsa_versions.py
│   ├── plot_trace_results.py
│   └── cultural_rule_validator.py
│
├── pedagorelearn_env.py           			 						# Core Gymnasium environment
├── tutor_train_sarsa_rewarded.py   								# Main training script
├── rules_loader.py                 									# Loads & validates YAML rules
├── student_model_sarsa.py         	 							# Simulated learner model
├── train_runner.py / train_eval.py 								# Run control and evaluation
│
├── requirements.txt                									# Dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Quick Start

```bash
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate       				# macOS/Linux
# .venv\Scripts\activate        				# Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run baseline or SARSA training
python tutor_train_sarsa_rewarded.py

# 4. Visualize or compare results
python plot_trace_results.py
python compare_sarsa_versions.py
```

Expected output:
```
Episode 500 | Avg reward = 118.6 | Steps-to-mastery = 47 | Accuracy = 0.91
Aggregated policy surpasses fixed curriculum baseline.
```

---

## Tech Stack

•	Python 3.10+
•	Gymnasium 0.29+
•	NumPy 1.23+
•	Matplotlib 3.7+
•	PyYAML for knowledge-base parsing

---

## Future Directions
•	Full aggregation ablation across four schemes
•	Sensitivity analysis for learning rate (α), discount (γ), and ε-decay
•	Integration of heuristic spaced-repetition baseline
•	Policy visualization and interpretability plots
•	Scaling to additional cultural domains and extended rulesets

---

## Acknowledgments

PedagoReLearn was developed as part of **CSCE 642: Reinforcement Learning (Fall 2025)** at **Texas A&M University**.  

Conceptual design and implementation by **Thomas F. Hallmark** and **Jun Kwon**.
>**AUTHOR BIOGRAPHIES**
>
> **Hallmark, T. F. (2025).**
>
>Thomas F. Hallmark is a doctoral student in Curriculum and Instruction with a cognate in Engineering Education in the Department of Teaching, Learning, and Culture at Texas A&M University. He holds degrees in Legal Studies and Business Administration (MBA) and brings more than 30 years of experience in the nuclear and utilities industries. His research focuses on the integration of artificial intelligence and reinforcement learning in engineering and STEM education, emphasizing adaptive tutoring systems, veteran transitions, and cross-cultural learning. Hallmark’s work combines pedagogical theory with computational modeling to design human-centered AI learning environments.
>
> **Kwon, J. (2025).**
>
>Jun Kwon is a graduate student in Computer Science and Engineering at Texas A&M University, specializing in machine learning and artificial intelligence applications for education and human-computer interaction. His research interests include reinforcement learning algorithms, neural network optimization, and adaptive feedback mechanisms in educational software. Kwon contributes to the computational architecture and algorithmic implementation of PedagoReLearn, focusing on model design, environment development, and performance evaluation across multiple RL frameworks.
> 
>> **Joint Contribution**
>Hallmark and Kwon collaboratively developed the conceptual framework and technical implementation of PedagoReLearn, merging educational theory and AI engineering to advance research in adaptive tutoring systems and cultural-learning reinforcement models.

## GitHub Description

Adaptive RL tutoring system modeling cultural learning through Dewey-inspired state, action, and reward design.

## AI Use Disclaimer

Artificial intelligence (AI) tools, including ChatGPT, were used to assist with grammar, formatting, and organization of this document. All intellectual content—including code, research design, and analysis—remains the original work of the authors. Use of AI assistance complies with Texas A&M University’s academic integrity guidelines and does not replace human authorship or scholarly contribution.

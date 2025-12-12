# PedagoReLearn


**© 2025 Thomas F. Hallmark**  
Licensed under the [MIT License](LICENSE)

---

## Overview

**PedagoReLearn** is an **AI-driven reinforcement learning (RL) framework** for adaptive cross-cultural tutoring.  
It models cultural competence training as a *state–action–reward* process where an agent learns when to **teach**, **review**, or **quiz** learners across domains such as etiquette, privacy, work, and travel.

Grounded in **John Dewey’s progressive education theory**, the system demonstrates how pedagogical strategies can *emerge autonomously through experience*, advancing long-term mastery and retention.

---

## Project Summary

PedagoReLearn formulates tutoring as a **Markov Decision Process (MDP)**:

| Component | Description |
|------------|-------------|
| **States** | Learner mastery levels and recency of review for each cultural rule |
| **Actions** | `teach`, `quiz`, or `review` |
| **Rewards** | Reflect learning success, retention, and teaching efficiency |

Each YAML file under `/rules/` defines a **cultural knowledge domain** (e.g., workplace etiquette, travel behavior, hygiene norms).  
The **Gymnasium environment** (`env.env.py`) interprets these as interactive learning topics.

---

##  Repository Structure
```
PedagoReLearn/
│	# -------------------- Documents --------------------
├── _archive/						# Archive of the whole semester
├── rules/							# YAML cultural knowledge base
│
│	#------------------    Codes    --------------------
├── agents/tutor.py                 # RL Tutor agents implementations. includes:
│			            			# 	SARSA(0) on-policy learner
│   					          	# 	Random baseline policy
│ 									#	Fixed baseline policy
├── env/env.py                      # Core Gymnasium environment
├── students/student.py             # Simulated Student
├── utils/                          # Utilities
│   ├── data_util.py				# 	Full result data structure & logger
│   └── plotter.py					# 	Full result figure generator
│
├── trainer.py                      # RL tutor trainer
├── main_train.py                   # Main script for train & result
│
│	#------------------   Results   --------------------
├── fig_results/                    # Figures of results
├── ts_results/                    	# Full results for each step in episode
│									# to generate figures
│
├── LICENSE
├── README.md
└── requirements.txt
```
---

## Quick Start

#### Installation

```bash
# 0. Clone repository

# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt
```
#### View Student model
```bash
# compare student's forgetting curve to Ebbinghaus Forgetting Curve
python -m students.student forgetting_curve

```

#### Train & Get result
```bash
# Train PedagoReLearn SARSA Tutor for n_seeds
# and compare with random and fixed policy tutor. 
python -m main_train --n_rules 3 --n_episodes 2000 --n_seeds 50 --ma_window 0.01 --track_full_result

```
Resulting figures can be found in fig_results/


### Tech Stack
- Python 3.10+
- Gymnasium ≥ 0.29
- NumPy ≥ 1.23
- Matplotlib ≥ 3.7
- PyYAML for rule parsing and validation

### Future Directions
- Full aggregation ablation across four schemes
- Sensitivity analysis for α, γ, and ε-decay parameters
- Integration of heuristic spaced-repetition baseline
- Policy interpretability visualization (heatmaps, frequency plots)
- Expansion to additional cultural domains and learner models
	
### Acknowledgments

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

**GitHub Description**

Adaptive RL tutoring system modeling cultural learning through Dewey-inspired state, action, and reward design.

**AI Use Disclaimer**

Artificial intelligence (AI) tools—including ChatGPT—were used only for grammar, formatting, and document organization.
All intellectual content (code, methodology, analysis) is the original work of the authors and complies with Texas A&M University academic integrity standards.



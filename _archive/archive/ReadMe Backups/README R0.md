# PedagoReLearn

**Copyright © 2025 Thomas F. Hallmark**  
Licensed under the MIT License (see [LICENSE](LICENSE)).

PedagoReLearn is an **AI-driven tutoring environment** that uses **Reinforcement Learning (RL)** to teach and adapt to learners’ progress across cultural knowledge domains (e.g., professional etiquette, digital privacy, transportation, hygiene).  

It serves as the foundation for exploring adaptive learning through *state–action–reward* modeling, where an AI tutor learns to optimize teaching and review sequences over time.

---

## Project Overview

PedagoReLearn models tutoring as a **Markov Decision Process (MDP)**:

- **States:** learner mastery levels and recency of review for each cultural rule  
- **Actions:** teach, quiz, review, or pause (no-op)  
- **Rewards:** reflect learning success, retention, and teaching efficiency  

Each YAML file under `/rules/` defines a domain of required cultural behaviors (e.g., workplace, travel, or hygiene norms).  
The RL environment (`pedagorelearn_env.py`) interprets these as learning topics.

---

## 📁 Repository Structure

```
PedagoReLearn/
│
├── pedagorelearn_env.py         # Gymnasium environment (Week 2 core)
├── demo_random_run.py           # Test runner for quick validation
│
├── student_model.py             # Learner simulation logic
├── tutor_env.py                 # Higher-level tutor agent (Week 3+)
├── cultural_rule_validator.py   # YAML rule parsing and integrity checks
│
├── rules/                       # YAML rule sets by category
│   ├── work_professional.yaml
│   ├── transport_travel.yaml
│   ├── digital_privacy.yaml
│   ├── religion_customs.yaml
│   ├── economy_society.yaml
│   ├── hygiene.yaml
│   ├── emergency_legal.yaml
│
├── docs/                        # Supporting documents
│   ├── Proposal - PedagoReLearn.pdf
│   ├── Outline R5 - Project Proposal (with Dewey).docx
│   └── README_German_Cultural_Rules_Handbook_Final.pdf
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run demo test
python demo_random_run.py
```

Expected output:
```
Action meanings: ['teach(0:rule_0)', 'quiz(0:rule_0)', 'review(0:rule_0)', ...]
Episode 0 ended at t=12 | total_reward=45.8 | mastered=True
Done.
```

---

## Tech Stack

- **Language:** Python 3.10+  
- **Core Library:** [Gymnasium](https://gymnasium.farama.org/)  
- **Other Dependencies:** `numpy`, `yaml`

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

AI tools were used for grammar and formatting assistance only; all conceptual work, code, and design remain original.

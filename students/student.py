import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Student model / transition dynamics ---
TEACH_SUCCESS             = (0.9, 0.6, 0.4, 0)          # P(level up) on TEACH (if < L3)
REVIEW_REINFORCE          = 0.20                        # P(level up) on REVIEW (if < L3)
QUIZ_BASE_P               = (0.1, 0.4, 0.7, 0.95)       # P(correct) at L1, L2, L3 (pre-spacing)
QUIZ_LVLUP_ON_CORRECT_P   = 0.30                        # P(level up) when quiz is correct (if < L3)
QUIZ_LVLDOWN_ON_WRONG_P    = 0.2                        # P(level down) when quiz is incorrect (if > L0)

class Student:
    MASTERY_MAX = 3
    # Forgetting model
    FORGET_RECENCY_THRESHOLD = 4
    FORGET_PROBS = (0.15, 0.15, 0.05)  # P(drop 1 level) from L1, L2, L3

    def __init__(self, n_rules: int, R_max: int = 30, seed: int | None = None):
        self.N = int(n_rules)
        self.R_max = int(R_max)
        self._rng = np.random.default_rng(seed)

        self._mastery = np.zeros(self.N, dtype=np.int32)
        self._recency = np.full(self.N, 2, dtype=np.int32)  # start with moderate gaps
        self._step = 0

    # --- Public API ---
    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mastery[:] = 0
        self._recency[:] = 2
        self._step = 0

    def instruct(self, i_type: str, rule: int|None):
        r = int(rule) if rule is not None else None
        self._step += 1
        response = None

        # Apply action dynamics (mirrors env student model)
        if i_type == 'teach':
            m = int(self._mastery[r])
            if self._rng.random() < TEACH_SUCCESS[m]:
                self._mastery[r] += 1
            self._recency[r] = 0

        elif i_type == 'review':
            if self._mastery[r] < self.MASTERY_MAX and self._rng.random() < REVIEW_REINFORCE:
                self._mastery[r] += 1
            self._recency[r] = 0

        elif i_type == 'quiz':
            m = int(self._mastery[r])
            correct = bool(self._rng.random() < QUIZ_BASE_P[m])
            if correct and self._rng.random() < QUIZ_LVLUP_ON_CORRECT_P:
                self._mastery[r] = min(m+1, self.MASTERY_MAX)
            elif not correct and self._rng.random() < QUIZ_LVLDOWN_ON_WRONG_P:
                self._mastery[r] = max(0,m-1)
            self._recency[r] = 0
            response = correct

        # Apply forgetting to items with large gaps
        mask = self._recency >= self.FORGET_RECENCY_THRESHOLD
        if np.any(mask):
            for i in np.where(mask)[0]:
                m = int(self._mastery[i])
                if m <= 0:
                    continue
                drop_idx = min(m - 1, len(self.FORGET_PROBS) - 1)
                if self._rng.random() < self.FORGET_PROBS[drop_idx]:
                    self._mastery[i] = m - 1

        # Time passes
        self._recency = np.minimum(self._recency + 1, self.R_max)

        return response

    def get_state(self):
        return {
            'mastery': self._mastery.copy(),
            'recency': self._recency.copy(),
            'step': np.int32(self._step),
        }

# ---- Student Analysis functions ----
def plot_forgetting_curve():
    """plot forgetting trend"""
    n_seeds = 10
    n_steps = 100
    # avg_mastery[seed,t] = average mastery over n_rules at time=t, seed=seed
    avg_mastery = np.zeros((n_seeds,n_steps))
    for seed in range(n_seeds):
        stu = Student(n_rules=3,seed=seed)
        stu.reset()
        stu._mastery = np.full_like(stu._mastery,stu.MASTERY_MAX)
        
        for t in range(n_steps):
            avg_mastery[seed,t] = np.mean(stu._mastery)
            stu.instruct('noop',None)

    # data to plot - normalize mastery
    avg_mastery /= stu.MASTERY_MAX
    # observed student forgetting curve
    x_values = np.arange(n_steps)
    scatter_points = (
        np.tile(x_values,n_seeds),
        avg_mastery.flatten()
    )
    avg_m_over_seeds = np.mean(avg_mastery,axis=0)
    
    # fit an Ebbinghaus forgetting curve to the observed
    reg = LinearRegression(fit_intercept=False)
    targets = -np.log(avg_m_over_seeds)
    reg.fit(x_values.reshape(-1,1),targets)
    lamb = reg.coef_[0]
    # print(lamb)
    
    # Ebbinghaus forgetting cuve
    e_curve = np.exp(-lamb*x_values)


    plt.figure(figsize=(10,6))
    plt.scatter(
        *scatter_points, color='C0',
        s=2, label='student model')
    plt.plot(
        x_values,avg_m_over_seeds, ls='--',  color='C1',
        label='average student model')
    plt.plot(
        x_values,e_curve, ls='-', color='C2',
        label=f'y=e^{{-{lamb:.3f}*t}}')


    plt.legend()
    plt.xlabel('Time since last review (t)')
    plt.ylabel('Retention probability')
    plt.title(f'Ebbinghaus Forgetting Curve vs Average Student Forgetting ({n_seeds} seeds)')
    # plt.show()
    plt.savefig("fig_results/student_forgetting.png")
    print("Figure saved as \"fig_results/student_forgetting.png\"")
    
def manual_interaction():
    stu = Student(n_rules=4,seed=27)
    while True:
        state = stu.get_state()
        print(state)
        try:
            i_type = input("instruct :")
            i_type = 'teach' if i_type == 't' else i_type
            i_type = 'quiz' if i_type == 'q' else i_type
            i_type = 'review' if i_type == 'r' else i_type
            
            rule_id = int(input("rule id :"))
        except ValueError:
            continue
        
        if i_type in ('teach','quiz','review'):
            if rule_id >=0 and rule_id < 5:
                ok = stu.instruct(i_type,rule_id)
                print(f"response {ok}")
        else: 
            stu.instruct('noop',1)

def test_teach():
    n_rules = 4
    n_tests = 100
    steps = []
    for seed in range(n_tests):
        stu = Student(n_rules,seed=seed)
        for i in range(200):
            rule = np.argmin(stu._mastery)
            _ = stu.instruct('teach',rule)
            if np.all(stu._mastery == stu.MASTERY_MAX):
                steps.append(i)
                break
        if np.any(stu._mastery < stu.MASTERY_MAX):
            steps.append(np.nan)
    
    print(f"min: {np.min(steps)}")
    print(f"mean: {np.mean(steps)}")
    print(f"std: {np.std(steps)}")
    print(f"max: {np.max(steps)}, {np.argmax(steps)}")

import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Simulated Student.")
    p.add_argument("command",choices=['forgetting_curve','interact','test_teach'])
    args = p.parse_args()

    if args.command == 'forgetting_curve':
        plot_forgetting_curve()

    elif args.command == 'interact':
        manual_interaction()
    
    elif args.command == 'test_teach':
        test_teach()

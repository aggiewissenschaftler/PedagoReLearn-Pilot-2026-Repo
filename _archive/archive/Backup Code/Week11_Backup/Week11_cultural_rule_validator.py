"""
cultural_rule_validator.py
--------------------------------
Validates user responses against cultural rules defined in a YAML file.

Usage:
    from cultural_rule_validator import CulturalRuleValidator

    validator = CulturalRuleValidator(yaml_path="cultural_rules_DE.yaml", culture="DE")
    score, feedback = validator.evaluate(rule_key="greetings_closings", user_text="Guten Tag, Frau Müller.")
    print(score, feedback)

Scoring:
- +1  : matches a 'correct' example/keyword and no 'incorrect' keyword
- -1  : matches an 'incorrect' keyword
-  0  : no clear match (neutral); encourage learning prompt

Notes:
- This is a lightweight keyword-based validator. For production, consider NLP (regexes, lemmatization).
- Searches both REQUIRED and SUGGESTED rule groups for the given rule_key.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import yaml
import re


@dataclass
class Rule:
    description: str
    correct: List[str]
    incorrect: List[str]


class CulturalRuleValidator:
    def __init__(self, yaml_path: str, culture: str = "DE"):
        """Load YAML rules for the specified culture."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if culture not in data:
            raise ValueError(f"Culture '{culture}' not found in YAML.")
        self.rules = self._flatten_rules(data[culture])
        self.culture = culture

    def _flatten_rules(self, culture_dict: Dict) -> Dict[str, Rule]:
        """Flatten REQUIRED and SUGGESTED rule groups into a single dict."""
        out: Dict[str, Rule] = {}
        for group in ("REQUIRED", "SUGGESTED"):
            if group not in culture_dict:
                continue
            for key, val in culture_dict[group].items():
                desc = val.get("description", "")
                correct = val.get("correct", []) or []
                incorrect = val.get("incorrect", []) or []
                out[key] = Rule(desc, list(correct), list(incorrect))
        return out

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace for robust matching."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def evaluate(self, rule_key: str, user_text: str) -> Tuple[int, str]:
        """
        Evaluate user_text against a specific rule.
        Returns (score, feedback).
        """
        if rule_key not in self.rules:
            return (0, f"Rule '{rule_key}' not found for culture {self.culture}.")

        rule = self.rules[rule_key]
        text_norm = self._normalize(user_text)

        # Keyword/phrase match against incorrect list first (negative wins)
        for bad in rule.incorrect:
            if not bad:
                continue
            if self._normalize(bad) in text_norm:
                return (-1, f"Careful: That clashes with '{rule_key}'. {rule.description}")

        # Then test for any 'correct' pattern
        for good in rule.correct:
            if not good:
                continue
            if self._normalize(good) in text_norm:
                return (+1, f"Nice! That aligns with '{rule_key}'. {rule.description}")

        # Neutral
        return (0, f"No clear match for '{rule_key}'. Hint: {rule.description}")

    def rubric(self, rule_key: str) -> Optional[Rule]:
        """Return the rule object (description + examples) for UI tooltips."""
        return self.rules.get(rule_key)

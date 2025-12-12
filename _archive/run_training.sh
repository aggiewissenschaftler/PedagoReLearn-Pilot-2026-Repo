#!/bin/zsh
# Generic script to run any weekly training and analysis in PedagoReLearn
# Usage: ./run_training.sh <trainer> [description]
# Example:
#   ./run_training.sh tutor_train_sarsa_rewarded.py "Week 6: Reward-Shaped SARSA"
#   ./run_training.sh tutor_train_sarsa.py "Week 5: Baseline SARSA"

TRAINER=${1:-tutor_train_sarsa_rewarded.py}
DESC=${2:-"Training Run"}

echo "🚀 Starting ${DESC} ..."
python "${TRAINER}"

echo ""
echo "📊 Running analysis on latest results..."
python analyze_csv.py --logs_dir trace_results/logs/

echo ""
echo "✅ ${DESC} complete! Check:"
echo "   → trace_results/logs/  for CSV logs"
echo "   → trace_results/figs/  for timestamped plots"
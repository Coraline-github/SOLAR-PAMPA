# SOLAR-PAMPA — Main Pipeline Orchestrator
# Usage:
#   python pipeline.py              → runs full pipeline
#   python pipeline.py features     → resumes from feature engineering
#   python pipeline.py modeling     → resumes from modeling
#   python pipeline.py dashboard    → only rebuilds dashboard

import sys
from pathlib import Path
from config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR

from agents.data_agent       import DataAgent
from agents.feature_agent    import FeatureAgent
from agents.modeling_agent   import ModelingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.dashboard_agent  import DashboardAgent

def run_pipeline(start_from: str = "data"):

    stages = {
        "data":       DataAgent(DATA_RAW, DATA_PROCESSED),
        "features":   FeatureAgent(DATA_PROCESSED, DATA_PROCESSED),
        "modeling":   ModelingAgent(DATA_PROCESSED, MODELS_DIR),
        "evaluation": EvaluationAgent(MODELS_DIR, REPORTS_DIR),
        "dashboard":  DashboardAgent(REPORTS_DIR, Path("dashboard")),
    }

    stage_order = list(stages.keys())

    if start_from not in stage_order:
        print(f"❌ Unknown stage: '{start_from}'")
        print(f"   Valid options: {stage_order}")
        sys.exit(1)

    start_idx = stage_order.index(start_from)

    print(f"\n🌞 SOLAR-PAMPA Pipeline")
    print(f"   Starting from : {start_from.upper()}")
    print(f"   Stages to run : {stage_order[start_idx:]}\n")

    for stage_name in stage_order[start_idx:]:
        agent   = stages[stage_name]
        print(f"▶  {stage_name.upper()} AGENT running...")

        success = agent.execute()

        if not success:
            print(f"\n❌ Pipeline stopped at : {stage_name}")
            print(f"   Fix the error above and resume with:")
            print(f"   python pipeline.py {stage_name}")
            sys.exit(1)

        print(f"✅ {stage_name.upper()} AGENT done\n")

    print("🎉 Pipeline complete!")
    print("   Launch dashboard: streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_pipeline(start_from=start)
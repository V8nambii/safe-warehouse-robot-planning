# Evaluating a Local LLM for Safe Warehouse Robot Planning

**Proposed title:** *Evaluating a Local LLM for Safe Warehouse Robot Planning*

This project extends the professor-supplied **Qwen robot simulator** into a reproducible MSc research system. It contains three original experiments and one exploratory extension:

1. **Direction reasoning:** the robot follows relative movement and turning instructions, then the LLM identifies the target's world compass direction.
2. **Route planning:** the LLM proposes a complete warehouse route, which is checked against obstacles and an optimal A* route.
3. **Task-and-route planning:** the LLM converts a worker's instruction into a structured pick-and-deliver job containing the correct item, destination and route. The route must visit an accessible pickup cell before reaching the requested dock.
4. **Exploratory hybrid planning:** the LLM extracts the requested action, pallet and dock, while the verified A* component creates and validates the physical route.

The project is Amazon-inspired, but it is not an Amazon product and does not reproduce Amazon's private systems. It is a safe 2D research simulator with no connection to a physical robot.

## What is included

- Corrected coordinate and orientation handling
- Reproducible scenario generation using fixed random seeds
- Four controlled difficulty levels
- Four direction-prompt designs and three route-prompt designs
- Qwen access through a local Ollama server
- Symbolic direction ground truth
- A* optimal-route ground truth
- Natural-language pick-and-deliver instructions
- Structured item/destination task validation
- Automatic pickup-access verification
- Automatic CSV and JSON result files
- Direction accuracy, strict accuracy and format compliance
- Route validity, collision, goal-reaching, optimality and efficiency metrics
- Automatic sample-integrity checks and inference-error counts
- Overall accuracy tables with numerators, denominators and 95% Wilson intervals
- Automatically generated brief accuracy report and overview graph
- Paired direct-LLM versus hybrid LLM+A* mission comparison
- Interactive Streamlit demonstration
- Automated tests
- Separate Google Colab and Mac/Jupyter experiment notebooks
- Experiment, dissertation, presentation and submission guidance

## Project structure

```text
warehouse-llm-dissertation/
├── warehouse_llm/                 Core research software
│   ├── simulator.py               Scenario generator and state transitions
│   ├── planner.py                 A* and route validation
│   ├── prompts.py                 Controlled prompt conditions
│   ├── llm.py                     Ollama and baseline clients
│   ├── evaluator.py               Experiment runner and metrics
│   └── visualization.py           Warehouse figures
├── tests/                          Automated verification
├── docs/                           Research and submission guides
├── Warehouse_LLM_Experiments_Colab.ipynb
├── Warehouse_LLM_Experiments_Mac.ipynb
├── streamlit_app.py               Interactive demonstration
├── analyse_results.py             Integrity checks, accuracy report, tables and figures
└── pyproject.toml                  Installation configuration
```

## Recommended route: Google Colab

1. Download the complete project ZIP.
2. Open `Warehouse_LLM_Experiments_Colab.ipynb` in Colab.
3. Select a GPU runtime if available.
4. Follow the notebook from top to bottom.
5. Run the oracle check first. It must achieve 100% because it verifies the software pipeline.
6. Run the small Qwen pilot and inspect its CSV files.
7. Run the final experiment without changing cases after seeing their answers.
8. Download the generated results ZIP and save the executed notebook.

## Mac and local Jupyter route

1. Extract the complete project ZIP in Downloads.
2. Install and open the Ollama application for macOS.
3. Start Jupyter and open `Warehouse_LLM_Experiments_Mac.ipynb` from inside the extracted project folder.
4. Run each notebook cell in order with **Shift + Enter**.
5. The Mac notebook finds the project folder and uses Jupyter's active Python interpreter automatically.
6. Follow [MAC_JUPYTER_GUIDE.md](docs/MAC_JUPYTER_GUIDE.md) if a checkpoint does not pass.

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Install Ollama separately, start it, and pull the same model used in the professor's notebook:

```bash
ollama serve
ollama pull qwen:7b
```

Run a small pilot:

```bash
python -m warehouse_llm.cli direction --client ollama --model qwen:7b --cases 1 --output-dir results/pilot
python -m warehouse_llm.cli route --client ollama --model qwen:7b --cases 1 --output-dir results/pilot
python -m warehouse_llm.cli mission --client ollama --model qwen:7b --cases 1 --output-dir results/pilot
```

Run the approved final design:

```bash
python -m warehouse_llm.cli direction --client ollama --model qwen:7b --cases 30 --seed 202600 --output-dir results/final
python -m warehouse_llm.cli route --client ollama --model qwen:7b --cases 20 --seed 202700 --output-dir results/final
python -m warehouse_llm.cli mission --client ollama --model qwen:7b --cases 20 --seed 202800 --output-dir results/final
```

Run the optional exploratory hybrid comparison separately:

```bash
python -m warehouse_llm.cli hybrid --client ollama --model qwen:7b --cases 20 --seed 202800 --output-dir results/hybrid
python analyse_results.py --input-dir results/final --hybrid-dir results/hybrid --output-dir results/analysis
```

Start the demonstration interface:

```bash
streamlit run streamlit_app.py
```

## Research safeguards

- `oracle-pipeline-check` is a software test, not an LLM result.
- `random-baseline` is a weak comparison baseline, not an LLM.
- Do not report Qwen results until the Qwen experiment has actually run.
- Do not delete failed responses, selectively regenerate difficult cases or manually edit the raw CSV.
- Record the model name, model digest/version, temperature, seeds, date, hardware and software versions.
- Read and understand the code before presenting it.
- Follow the University's current rules for declaring AI assistance and third-party starter code.

Start with [STEP_BY_STEP_CODING_COURSE.md](docs/STEP_BY_STEP_CODING_COURSE.md), then follow [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md), [TESTING_MANUAL.md](docs/TESTING_MANUAL.md) and [FINAL_SUBMISSION_CHECKLIST.md](docs/FINAL_SUBMISSION_CHECKLIST.md).

For exact metric definitions and the generated report files, read
[ACCURACY_REPORT_GUIDE.md](docs/ACCURACY_REPORT_GUIDE.md).
For the optional follow-up comparison, read
[HYBRID_EXPERIMENT_GUIDE.md](docs/HYBRID_EXPERIMENT_GUIDE.md).

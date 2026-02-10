# To setup
pip install -r requirements.txt
Either create a .env file which has: 
export DB_PASS="Damn123,"
export OPENAI_API_KEY=".."
export GOOGLE_API_KEY=".."

Run setup_aemab.sh after adapting it to your repository structure. This will uncompress the solver and place it in your solvers/base directory, as well as removing the broken symlink.

# Data Generation:

## generation
To use, run: PYTHONPATH=src python src/llmsat/pipelines/gemini_data_generation.py

In the main function of gemini_data_generation.py you can configure the number of team leaders and team members and which prompts you are using. The script currently creates the leaders and the variants (mutants) and generates the code for all of them. 

(If you are missing llmsat module, you need to prepend python command with PYTHONPATH=src)

## evaluation
To use, run: PYTHONPATH=src python src/llmsat/pipelines/evaluation.py --run_all --generation_tag {your tag} --build-only
The build-only flag uses the function_registry.yaml file to inject the generated codes into their own versions of the solver, as you can see in solvers/{algorithm_...}




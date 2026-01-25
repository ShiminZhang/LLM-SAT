# To setup
pip install -r requirements.txt
Set up your database password and openai api key and PYTHONPATH. 
Either create a .env file which has: 
export DB_PASS="Damn123,"
export OPENAI_API_KEY=".."

or source the export scripts.

Run setup_aemab.sh after adapting it to your repository structure. This will uncompress the solver and place it in your solvers/base directory.

# ChatGPT pipeine:

## generation
to use, run: python src/llmsat/pipelines/chatgpt_data_generation.py

In the main function of chatgpt_data_generation.py you can configure the number of team leaders and team members and which prompts you are using. The script currently creates the leaders and the variants (mutants) and generates the code for all of them. 




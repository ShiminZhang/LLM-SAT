import argparse
import json
from tqdm import tqdm

def create_test_data():
    base_data = [
        {
            "prompt": "Hello, how are you?",
            "chosen": "I'm doing well, thank you for asking! How can I help you today?",
            "rejected": "I don't know."
        },
        {
            "prompt": "What is Python?",
            "chosen": "Python is a high-level programming language known for its simplicity and readability.",
            "rejected": "Python is a snake."
        },
        {
            "prompt": "Explain machine learning",
            "chosen": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "rejected": "Machine learning is when machines learn by themselves."
        },
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "How does photosynthesis work?",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
            "rejected": "Plants just grow somehow."
        },
        {
            "prompt": "What is quantum computing?",
            "chosen": "Quantum computing is a type of computation that uses quantum mechanical phenomena like superposition and entanglement to process information.",
            "rejected": "It's just faster computers."
        },
        {
            "prompt": "Explain the water cycle",
            "chosen": "The water cycle is the continuous movement of water through evaporation, condensation, and precipitation processes.",
            "rejected": "Water goes up and comes down."
        },
        {
            "prompt": "What is democracy?",
            "chosen": "Democracy is a system of government where power is vested in the people, who rule either directly or through elected representatives.",
            "rejected": "It's when people vote."
        },
        {
            "prompt": "How do solar panels work?",
            "chosen": "Solar panels convert sunlight into electricity using photovoltaic cells that create an electric current when exposed to light.",
            "rejected": "They just collect sunlight."
        },
        {
            "prompt": "What is climate change?",
            "chosen": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
            "rejected": "The weather is changing."
        },
        {
            "prompt": "Explain artificial intelligence",
            "chosen": "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            "rejected": "It's when computers are smart."
        },
        {
            "prompt": "What is blockchain?",
            "chosen": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records secured using cryptography.",
            "rejected": "It's just a database."
        },
        {
            "prompt": "How do vaccines work?",
            "chosen": "Vaccines work by introducing a weakened or inactive form of a pathogen to stimulate the immune system to produce antibodies.",
            "rejected": "They just prevent diseases."
        },
        {
            "prompt": "What is renewable energy?",
            "chosen": "Renewable energy comes from natural sources that are constantly replenished, such as solar, wind, and hydroelectric power.",
            "rejected": "It's energy that doesn't run out."
        },
        {
            "prompt": "Explain the theory of evolution",
            "chosen": "The theory of evolution explains how species change over time through natural selection, genetic variation, and adaptation to environmental pressures.",
            "rejected": "It's about how animals change."
        },
        {
            "prompt": "What is the capital of China?",
            "chosen": "The capital of China is Beijing, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Japan?",
            "chosen": "The capital of Japan is Tokyo, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Korea?",
            "chosen": "The capital of Korea is Seoul, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of India?",
            "chosen": "The capital of India is New Delhi, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Germany?",
            "chosen": "The capital of Germany is Berlin, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Italy?",
            "chosen": "The capital of Italy is Rome, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Spain?",
            "chosen": "The capital of Spain is Madrid, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        
        {
            "prompt": "What is the capital of Portugal?",
            "chosen": "The capital of Portugal is Lisbon, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Greece?",
            "chosen": "The capital of Greece is Athens, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },  
        {
            "prompt": "What is the capital of Turkey?",
            "chosen": "The capital of Turkey is Ankara, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Egypt?",
            "chosen": "The capital of Egypt is Cairo, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },  
        {
            "prompt": "What is the capital of Nigeria?",
            "chosen": "The capital of Nigeria is Abuja, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of South Africa?",
            "chosen": "The capital of South Africa is Pretoria, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Brazil?",
            "chosen": "The capital of Brazil is Brasília, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Argentina?",
            "chosen": "The capital of Argentina is Buenos Aires, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Canada?",
            "chosen": "The capital of Canada is Ottawa, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Australia?",
            "chosen": "The capital of Australia is Canberra, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of New Zealand?",
            "chosen": "The capital of New Zealand is Wellington, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of South Korea?",
            "chosen": "The capital of South Korea is Seoul, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Thailand?",
            "chosen": "The capital of Thailand is Bangkok, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Vietnam?",
            "chosen": "The capital of Vietnam is Hanoi, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Malaysia?",
            "chosen": "The capital of Malaysia is Kuala Lumpur, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Indonesia?",
            "chosen": "The capital of Indonesia is Jakarta, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Philippines?",
            "chosen": "The capital of Philippines is Manila, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Singapore?",
            "chosen": "The capital of Singapore is Singapore, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
        {
            "prompt": "What is the capital of Hong Kong?",
            "chosen": "The capital of Hong Kong is Hong Kong, a beautiful city known for its art, culture, and history.",
            "rejected": "I don't know the capital."
        },
    ]
    return base_data

class DataManager:
    def __init__(self, data_path="/pscratch/sd/s/shimin/verltest/dataset/dataset_preprocessed.jsonl"):
        # self.data = create_test_data()
        self.data = []
        self.data_path = data_path
        self.load_data(data_path)
        pass


    def load_data(self, data_path):
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def get_data(self, size):
        if size is None:
            return self.data
        else:
            if size > len(self.data):
                self.data = self.data * (size // len(self.data) + 1)
            return self.data[:size]

def preprocess_data(original_data_point: dict):
    N_of_heuristic = len(original_data_point["prompts"].keys())
    N_for_winner = len(original_data_point["winner_solver"].keys())
    N_for_loser = len(original_data_point["loser_solver"].keys())
    N_of_heuristic = min(N_of_heuristic, N_for_winner, N_for_loser)
    data = []
    for index in range(N_of_heuristic):

        decomposed_data_point = {
            "prompt" : None,
            "chosen" : None,
            "rejected" : None
        }
        system_prompt = original_data_point["prompts"][str(index + 1)][0]["content"]
        user_prompt = original_data_point["prompts"][str(index + 1)][1]["content"]
        decomposed_data_point["prompt"] = f"{system_prompt}\n{user_prompt}"
        decomposed_data_point["chosen"] = original_data_point["winner_solver"][str(index + 1)]
        decomposed_data_point["rejected"] = original_data_point["loser_solver"][str(index + 1)]
        data.append(decomposed_data_point)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/dataset_raw.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    # data = DataManager(args.data_path).get_data(args.limit)
    if args.preprocess:
        with open(args.data_path, "r") as f:
            data = json.load(f)
            dpo_datas = []
            count = 0
            for data_point in tqdm(data):
                dpo_data =preprocess_data(data_point)
                dpo_datas.append(dpo_data)
                count += 1
                if count >= args.limit:
                    break
            with open("dataset/dataset_preprocessed.jsonl", "w") as f:
                for data_points in dpo_datas:
                    for data_point in data_points:
                        json.dump(data_point, f)
                        f.write("\n")

    if args.test:
        data = []
        with open("dataset/dataset_preprocessed.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
                break
            print(len(data[0]))
            print(data[0]["prompt"])
            print(data[0]["chosen"])
            print(data[0]["rejected"])

if __name__ == "__main__":
    main()
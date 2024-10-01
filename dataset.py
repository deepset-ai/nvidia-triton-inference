# The gensentence.py script
# Run as python3.10 genstence.py > /home/a30user/datasets/passages_eng_200.jsonl

import random
import json
import sys

random.seed(42)  # For reproducibility

# Parameters for the dataset
num_samples = 100  # Number of samples to generate
mean_tokens = 20  # Target mean token size
std_dev = 0  # Standard deviation for token length

# Predefined sentence templates with placeholders
sentence_templates = [
    "The {noun} {verb} over the {adjective} {noun}.",
    "A {adjective} {noun} {verb} quickly through the {noun}.",
    "She {verb} the {adjective} {noun} with great {noun}.",
    "In the {noun}, the {adjective} {noun} {verb} slowly.",
    "They {verb} the {noun} before {verb} to the {noun}.",
    "The {adjective} {noun} {verb} near the {adjective} {noun}.",
]

# Word lists
nouns = ["cat", "dog", "car", "tree", "house", "book", "river", "sky", "mountain", "ocean"]
verbs = ["jumps", "runs", "flies", "writes", "builds", "grows", "flows", "sings", "paints", "drives"]
adjectives = ["blue", "happy", "quick", "bright", "dark", "calm", "tall", "strong", "beautiful", "silent"]

# Function to generate a meaningful sentence based on the templates
def generate_meaningful_sentence():
    template = random.choice(sentence_templates)
    return template.format(
        noun=random.choice(nouns),
        verb=random.choice(verbs),
        adjective=random.choice(adjectives)
    )

# Function to generate random text with a given token length
def generate_random_text(token_length):
    sentence = []
    total_tokens = 0

    while total_tokens < token_length:
        current_sentence = generate_meaningful_sentence()
        sentence_tokens = len(current_sentence.split())
        if total_tokens + sentence_tokens > token_length:
            break
        sentence.append(current_sentence)
        total_tokens += sentence_tokens

    return ' '.join(sentence)

#  Old code being commented out
#    sentence = []
#    while len(sentence) < token_length:
#        sentence.append(generate_meaningful_sentence())
#    return ' '.join(sentence)[:token_length]


# Generate the dataset
dataset = []
for _ in range(num_samples):
    # Randomly determine the token length based on a normal distribution
    token_length = int(random.gauss(mean_tokens, std_dev))
    token_length = max(1, token_length)  # Ensure token length is positive
    text = generate_random_text(token_length)
    text = " ".join([text, "(Tokens:", str(len(text.split())),")"])
    
    # Append to the dataset as a JSON object (assuming text field)
    dataset.append({"text": text})

# Save the dataset to a .jsonl file
file_name="sample_data.jsonl";
if (len(sys.argv) > 1):
    file_name=sys.argv[1];

with open(file_name, "w") as f:
    for data in dataset:
        f.write(json.dumps(data) + "\n")

print(f"Generated {num_samples} samples with mean token size {mean_tokens} and standard deviation {std_dev}.")



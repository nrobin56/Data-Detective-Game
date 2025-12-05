import random
import math

# -------------------------------------------
# CELEBRITY DATASET
# -------------------------------------------
CELEBRITIES = [
    {"name": "Beyoncé",         "profession": "musician", "fame": "high",   "era": "2000s",  "label": 0},
    {"name": "Taylor Swift",    "profession": "musician", "fame": "high",   "era": "2010s",  "label": 0},
    {"name": "Denzel Washington","profession": "actor",   "fame": "high",   "era": "90s",    "label": 0},
    {"name": "Zendaya",         "profession": "actor",    "fame": "medium", "era": "2020s",  "label": 0},
    {"name": "Michael Jordan",  "profession": "athlete",  "fame": "high",   "era": "90s",    "label": 1},
    {"name": "LeBron James",    "profession": "athlete",  "fame": "high",   "era": "2000s",  "label": 1},
    {"name": "Simone Biles",    "profession": "athlete",  "fame": "medium", "era": "2010s",  "label": 1},
    {"name": "Ariana Grande",   "profession": "musician", "fame": "medium", "era": "2010s",  "label": 0},
    {"name": "Tom Holland",     "profession": "actor",    "fame": "medium", "era": "2020s",  "label": 0},
    {"name": "Serena Williams", "profession": "athlete",  "fame": "high",   "era": "2000s",  "label": 1},
]

FEATURES = ["profession", "fame", "era"]


# -------------------------------------------
# ENTROPY
# -------------------------------------------
def entropy(subset):
    label_counts = {}
    for item in subset:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1

    total = len(subset)
    ent = 0
    for count in label_counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


# -------------------------------------------
# SPLITTING & INFORMATION GAIN
# -------------------------------------------
def split_by_feature(dataset, feature):
    splits = {}
    for item in dataset:
        value = item[feature]
        if value not in splits:
            splits[value] = []
        splits[value].append(item)
    return splits


def information_gain(dataset, feature):
    total_entropy = entropy(dataset)
    splits = split_by_feature(dataset, feature)

    weighted_entropy = 0
    for subset in splits.values():
        weighted_entropy += (len(subset) / len(dataset)) * entropy(subset)

    return total_entropy - weighted_entropy


# -------------------------------------------
# SIMPLE DECISION TREE (ONE FEATURE)
# -------------------------------------------
def build_decision_tree(dataset):
    gains = {f: information_gain(dataset, f) for f in FEATURES}
    best_feature = max(gains, key=gains.get)
    return best_feature


def predict(tree_feature, item):
    # Majority label among samples sharing this feature value
    splits = split_by_feature(CELEBRITIES, tree_feature)
    group = splits[item[tree_feature]]

    labels = [x["label"] for x in group]
    return max(set(labels), key=labels.count)


# -------------------------------------------
# GAME LOOP
# -------------------------------------------
def play_game():
    print("\n🎬 WELCOME TO CELEBRITY DETECTIVE!")
    print("Try to guess whether the mystery celebrity is from ENTERTAINMENT (0) or SPORTS (1).")
    print("You may ask yes/no questions about: profession, fame, or era.")
    print("-" * 60)

    mystery = random.choice(CELEBRITIES)
    best_feature = build_decision_tree(CELEBRITIES)

    possible_questions = FEATURES.copy()

    while True:
        print("\nAsk a question or type 'guess':")
        print("Available questions:", possible_questions)

        choice = input("> ").strip().lower()

        if choice == "guess":
            guess_label = int(input("Your guess (0 = entertainment, 1 = sports): "))
            real_label = mystery["label"]

            print(f"\nThe mystery celebrity was: ⭐ {mystery['name']} ⭐")

            if guess_label == real_label:
                print("🟢 You guessed correctly!")
            else:
                print(f"🔴 Wrong! Their true category was: {real_label}")

            ai_pred = predict(best_feature, mystery)
            print(f"\n🤖 AI used feature '{best_feature}' and predicted: {ai_pred}")

            if ai_pred == real_label:
                print("👉 The AI also guessed correctly!")
            else:
                print("👉 The AI guessed wrong this time!")

            print("\nGame Over.\n")
            break

        elif choice in possible_questions:
            value = mystery[choice]
            print(f"Answer: YES — the celebrity's {choice} is '{value}'.")
            possible_questions.remove(choice)

        else:
            print("Invalid question. Try again.")


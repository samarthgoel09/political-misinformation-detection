"""
Sample data generator for development and testing.
Generates synthetic labeled text data mimicking Fakeddit's multi-class structure.
"""

import pandas as pd
import numpy as np
import random


# Label schemes matching Fakeddit
LABELS_2WAY = ["true", "fake"]
LABELS_3WAY = ["true", "satire/parody", "fake"]
LABELS_6WAY = [
    "true",
    "satire/parody",
    "misleading content",
    "imposter content",
    "false connection",
    "manipulated content",
]

# Template sentences by category
TEMPLATES = {
    "true": [
        "The Senate passed a new infrastructure bill with bipartisan support on Tuesday.",
        "According to official reports, unemployment rates fell to 3.5 percent this quarter.",
        "The Federal Reserve announced a 0.25 percent interest rate increase today.",
        "The president signed an executive order on climate policy during a ceremony at the White House.",
        "Voter turnout in the midterm elections reached 50 percent, according to election officials.",
        "The Supreme Court agreed to hear arguments on the immigration policy case next month.",
        "A bipartisan committee released its findings on government spending waste.",
        "Officials confirmed the trade agreement will take effect starting January 1.",
        "The Department of Education released new guidelines for student loan forgiveness.",
        "Congressional leaders from both parties agreed on a temporary spending measure.",
    ],
    "satire/parody": [
        "Area congressman reportedly shocked to learn his constituents actually read his tweets.",
        "Local politician promises to fix potholes by simply renaming them 'freedom craters'.",
        "Man who gets all his news from memes somehow perfectly informed about geopolitics.",
        "Senate votes unanimously to extend lunch break by 45 minutes in historic bipartisan move.",
        "Presidential candidate reveals bold plan to solve all problems by tweeting about them.",
        "Breaking: Congress discovers that laws work better when people actually follow them.",
        "Politician's fact-check rating so low it actually wraps around to being impressive.",
        "Study finds 90 percent of campaign promises were made with fingers crossed behind back.",
        "Town hall meeting descends into chaos when politician accidentally tells the truth.",
        "New bill would require all political ads to include laugh track.",
    ],
    "misleading content": [
        "EXPOSED: The hidden truth about government surveillance that they don't want you to know!",
        "This one weird trick is what politicians use to stay in power forever.",
        "Scientists CONFIRM what we've been saying all along about the economy.",
        "Exposed: The REAL reason behind the latest policy change will shock you.",
        "What the mainstream media won't tell you about the new healthcare law.",
        "EXPOSED: Secret meetings between officials reveal the TRUTH about spending.",
        "You won't believe what this leaked document says about the election.",
        "The government doesn't want you to know this about your tax dollars.",
        "BREAKING: whistleblower reveals the shocking truth behind the policy.",
        "This controversial study proves everything you thought about politics was WRONG.",
    ],
    "imposter content": [
        "Official White House statement: President declares national emergency over social media usage.",
        "AP Breaking News: Congress votes to abolish all federal holidays starting next year.",
        "Reuters Exclusive: Secret documents reveal plan to merge three government agencies.",
        "CDC Official Report: New study links political debates to increased stress hormones.",
        "FBI Press Release: Agency announces new division dedicated to monitoring meme accounts.",
        "Pentagon confirms development of weather control satellite for agricultural purposes.",
        "WHO declares political misinformation a global health emergency in surprise announcement.",
        "Official Treasury Department memo leaked revealing plans to redesign all currency.",
        "NASA Administrator statement: Agency to redirect funds from space to political education.",
        "FEMA announces new preparedness category for surviving election seasons.",
    ],
    "false connection": [
        "President caught in massive scandal (article about routine policy meeting).",
        "BREAKING: Government in crisis after shocking revelation (story about minor budget adjustment).",
        "Millions affected by new law (article about obscure regulatory change).",
        "Democracy under threat: dramatic headline for a routine committee hearing update.",
        "Economy on the brink of collapse says expert (article quotes one minor analyst).",
        "Outrage erupts over political decision (three people tweeted about it).",
        "Bombshell report rocks Washington (small footnote in a quarterly review).",
        "Historic moment as politician makes unprecedented move (routine procedural vote).",
        "Nation divided over controversial new policy (poll shows 52-48 split on minor issue).",
        "Crisis at the border reaches new heights (seasonal fluctuation in crossing numbers).",
    ],
    "manipulated content": [
        "The senator clearly stated support for abolishing all public schools in this speech.",
        "Doctored transcript shows the official admitting to corruption during the hearing.",
        "Edited video proves the politician made racist remarks at last night's rally.",
        "Altered quote from the governor shows plans to raise taxes by 500 percent.",
        "Manipulated photo shows the mayor at a controversial protest he never attended.",
        "Modified audio clip appears to show congressman accepting bribes from lobbyists.",
        "Cropped screenshot of politician's email taken completely out of context to suggest wrongdoing.",
        "Selectively edited interview makes it appear the candidate supports extremist views.",
        "Fake transcript of closed-door meeting circulated to undermine policy negotiations.",
        "Spliced together clips from different speeches to fabricate incriminating statement.",
    ],
}


def generate_sample_data(
    n_samples: int = 2000,
    label_scheme: str = "6way",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic labeled data for development and testing.

    Args:
        n_samples: Number of samples to generate.
        label_scheme: One of '2way', '3way', '6way'.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with 'text', 'label', and 'label_id' columns.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    if label_scheme == "2way":
        label_map = {
            "true": "true",
            "satire/parody": "fake",
            "misleading content": "fake",
            "imposter content": "fake",
            "false connection": "fake",
            "manipulated content": "fake",
        }
        labels_list = LABELS_2WAY
    elif label_scheme == "3way":
        label_map = {
            "true": "true",
            "satire/parody": "satire/parody",
            "misleading content": "fake",
            "imposter content": "fake",
            "false connection": "fake",
            "manipulated content": "fake",
        }
        labels_list = LABELS_3WAY
    else:  # 6way
        label_map = {k: k for k in LABELS_6WAY}
        labels_list = LABELS_6WAY

    texts = []
    labels = []

    samples_per_category = n_samples // len(TEMPLATES)

    for category, category_templates in TEMPLATES.items():
        for _ in range(samples_per_category):
            # Pick a random template and add slight variations
            template = random.choice(category_templates)

            # Add random noise/variation
            variations = [
                template,
                template + " " + random.choice(["Read more.", "Share this.", "Developing story.", ""]),
                random.choice(["BREAKING: ", "UPDATE: ", "JUST IN: ", ""]) + template,
            ]
            text = random.choice(variations)
            mapped_label = label_map[category]
            texts.append(text)
            labels.append(mapped_label)

    df = pd.DataFrame({"text": texts, "label": labels})

    # Assign numeric label IDs
    label_to_id = {label: idx for idx, label in enumerate(labels_list)}
    df["label_id"] = df["label"].map(label_to_id)

    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Trim to exact requested size
    if len(df) > n_samples:
        df = df.head(n_samples)

    print(f"Generated {len(df)} samples with {label_scheme} label scheme")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    df = generate_sample_data(n_samples=2000, label_scheme="6way")
    df.to_csv("data/sample_dataset.csv", index=False)
    print(f"\nSaved sample dataset to data/sample_dataset.csv")
    print(f"\nSample rows:")
    print(df.head(10).to_string())

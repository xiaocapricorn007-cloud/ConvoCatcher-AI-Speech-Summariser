# Install dependencies (Colab-friendly, lightweight models throughout)
!pip install -q -U transformers datasets evaluate accelerate sentencepiece pyarrow \
    keybert sentence-transformers scikit-learn seqeval

import os
import re
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, f1_score
)

# ---- Reproducibility -------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE_DEVICE = 0 if DEVICE == "cuda" else -1
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. Training will still run but will be considerably "
          "slower. In Colab: Runtime > Change runtime type > T4 GPU.")

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    def box(x, y, w, h, text, color="#4C72B0", fontsize=10, textcolor="white"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2, edgecolor="#2b2b2b", facecolor=color
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, color=textcolor, wrap=True, weight="bold")
        return (x + w / 2, y, x + w / 2, y + h)  # bottom-center, top-center points

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))

    # Stage 1: raw input
    _, top1 = box(3.0, 12.0, 4.0, 0.8, "Raw Customer Conversation", color="#55606e")[2:]
    b1_bottom = (5.0, 12.0)

    # Stage 2: cleaning
    b2_top = box(3.0, 10.7, 4.0, 0.8, "Text Cleaning & Preprocessing", color="#55606e")
    arrow(5.0, 12.0, 5.0, 11.5)

    # Stage 3: tokenizer
    box(3.0, 9.4, 4.0, 0.8, "Transformer Tokenizer\n(DistilBERT WordPiece)", color="#55606e")
    arrow(5.0, 10.7, 5.0, 10.2)

    # connector down to branch point
    arrow(5.0, 9.4, 5.0, 8.85)
    ax.plot([1.2, 8.8], [8.85, 8.85], color="#333333", lw=1.6)

    # Stage 4: four parallel heads
    heads = [
        (0.3, "Intent\nClassification\n(DistilBERT,\nfine-tuned)", "#c44e52"),
        (2.85, "Sentiment\nAnalysis\n(RoBERTa,\npretrained)", "#dd8452"),
        (5.4, "Urgency /\nPriority\n(DistilBERT,\nweak-supervised)", "#937860"),
        (7.95, "Entities / Topics\n(BERT-NER +\nKeyBERT + regex)", "#8172b2"),
    ]
    head_tops = []
    for x, label, color in heads:
        ax.plot([x + 0.85, x + 0.85], [8.85, 8.35], color="#333333", lw=1.6)
        box(x, 7.5, 1.9, 0.85, label, color=color, fontsize=8.3)
        head_tops.append((x + 0.95, 7.5))

    # converge into summarizer
    ax.plot([1.2, 8.8], [7.1, 7.1], color="#333333", lw=1.6)
    for x, _ in head_tops:
        ax.plot([x, x], [7.5, 7.1], color="#333333", lw=1.6)
    arrow(5.0, 7.1, 5.0, 6.55)

    # Stage 5: summarization
    box(3.0, 5.7, 4.0, 0.85, "Transformer Summarization\n(DistilBART fine-tuned on SAMSum)", color="#55606e", fontsize=9)
    arrow(5.0, 5.7, 5.0, 5.2)

    # Stage 6: structured output
    box(2.4, 3.9, 5.2, 1.0, "Structured Customer Intelligence Output\n{intent, sentiment, urgency, entities, topics, summary}",
        color="#2ca02c", fontsize=9.5)

    ax.set_title("End-to-End Conversational Intelligence Pipeline", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    plt.show()

draw_architecture_diagram()

import requests
import pandas as pd

DATASET_ID = "PolyAI/banking77"

# Canonical 77 intent label names, in the same order as the dataset's ClassLabel
# feature (id -> name). Hardcoded as a robust fallback / cross-check since the
# label-name metadata isn't always reachable (see notes below).
BANKING77_LABEL_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def load_banking77_via_pr_parquet(dataset_id: str = DATASET_ID):
    """Load Banking77 straight from its Parquet files.

    As of this writing, `PolyAI/banking77`'s `main` branch still ships the
    OLD-style Python loading script (`banking77.py`) -- which the current
    `datasets` library refuses to run at all ("Dataset scripts are no longer
    supported"), and which also breaks the `datasets-server` API's automatic
    Parquet conversion for the same reason (ConfigNamesError).
    A pending, not-yet-merged pull request on the repo (currently `refs/pr/7`,
    with `refs/pr/6` and a specific historical commit as older duplicates)
    already deleted the script and added plain Parquet files directly. Hugging
    Face lets you resolve files from an open PR branch the same way as `main`,
    so we pull from there -- no loading script, no `datasets` library version
    sensitivity, no dependency on `datasets-server` at all.
    Once that PR is merged into `main`, the "main" candidate below will start
    working too and will be tried first automatically.
    """
    candidate_revisions = [
        "main",                                        # will work once the PR merges
        "refs/pr/7",                                    # current open PR with parquet files
        "refs/pr/6",                                     # earlier duplicate PR
        "796a4623935746f71378f0ebd435635a8ce08e50",       # pinned commit, extra safety net
    ]
    last_err = None
    for revision in candidate_revisions:
        base = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/data"
        try:
            train_df = pd.read_parquet(f"{base}/train-00000-of-00001.parquet")
            test_df = pd.read_parquet(f"{base}/test-00000-of-00001.parquet")
            return train_df, test_df, revision
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not fetch Banking77 Parquet files from any known revision "
        f"(tried {candidate_revisions})."
    ) from last_err


def load_banking77_via_datasets_server(dataset_id: str = DATASET_ID):
    """Fallback 1: read the auto-converted Parquet files via the datasets-server API."""
    parquet_resp = requests.get(
        "https://datasets-server.huggingface.co/parquet", params={"dataset": dataset_id}, timeout=30
    )
    parquet_resp.raise_for_status()
    parquet_files = parquet_resp.json()["parquet_files"]

    def url_for(split):
        matches = [f["url"] for f in parquet_files if f["split"] == split]
        if not matches:
            raise ValueError(f"No parquet file found for split '{split}'")
        return matches[0]

    train_df = pd.read_parquet(url_for("train"))
    test_df = pd.read_parquet(url_for("test"))
    return train_df, test_df


def load_banking77_via_datasets_lib(dataset_id: str = DATASET_ID):
    """Fallback 2: the standard `datasets` library loader. Currently expected to fail
    for this dataset until the Parquet-conversion PR is merged into `main`, since the
    library refuses to execute the legacy `banking77.py` loading script."""
    from datasets import load_dataset
    raw = load_dataset(dataset_id)
    train_df = raw["train"].to_pandas()
    test_df = raw["test"].to_pandas()
    label_names = raw["train"].features["label"].names
    return train_df, test_df, label_names


try:
    train_df_full, test_df, used_revision = load_banking77_via_pr_parquet()
    label_names = BANKING77_LABEL_NAMES
    print(f"Loaded Banking77 via direct Parquet files (revision: {used_revision}).")
except Exception as e1:
    print(f"Direct Parquet approach failed ({type(e1).__name__}: {e1}); trying datasets-server...")
    try:
        train_df_full, test_df = load_banking77_via_datasets_server()
        label_names = BANKING77_LABEL_NAMES
        print("Loaded Banking77 via the Hugging Face datasets-server Parquet API.")
    except Exception as e2:
        print(f"datasets-server approach failed ({type(e2).__name__}: {e2}); falling back to `datasets.load_dataset`...")
        train_df_full, test_df, label_names = load_banking77_via_datasets_lib()
        print("Loaded Banking77 via `datasets.load_dataset` fallback.")

NUM_INTENT_LABELS = len(label_names)
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

train_df_full["intent"] = train_df_full["label"].map(id2label)
test_df["intent"] = test_df["label"].map(id2label)

print(f"Train examples : {len(train_df_full)}")
print(f"Test examples  : {len(test_df)}")
print(f"Num intents    : {NUM_INTENT_LABELS}")
train_df_full.head()
plt.figure(figsize=(11, 7))
top_intents = train_df_full["intent"].value_counts().nlargest(20)
sns.barplot(x=top_intents.values, y=top_intents.index, palette="viridis")
plt.title("Top 20 Most Frequent Intents (Training Set)")
plt.xlabel("Number of examples")
plt.ylabel("Intent")
plt.tight_layout()
plt.show()

counts = train_df_full["intent"].value_counts()
print(f"Most frequent intent : {counts.idxmax()}  ({counts.max()} examples)")
print(f"Least frequent intent: {counts.idxmin()}  ({counts.min()} examples)")
print(f"Mean examples/intent : {counts.mean():.1f}   Std: {counts.std():.1f}")

train_df_full["word_count"] = train_df_full["text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(train_df_full["word_count"], bins=25, kde=True, color="#4C72B0")
plt.title("Distribution of Utterance Length (Training Set)")
plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print(train_df_full["word_count"].describe())
print(f"\n99th percentile word count: {train_df_full['word_count'].quantile(0.99):.0f}")

def clean_text(text: str) -> str:
    """Light-touch cleaning: normalize whitespace/URLs, keep casing & punctuation."""
    text = str(text).strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)              # strip URLs
    text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\'\-\:\$\%]", " ", text)  # drop unusual symbols
    text = re.sub(r"\s+", " ", text).strip()                    # collapse whitespace
    return text

train_df_full["clean_text"] = train_df_full["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

train_df_full[["text", "clean_text"]].sample(5, random_state=SEED)

train_df, val_df = train_test_split(
    train_df_full,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df_full["label"],
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Train: {len(train_df)}   Val: {len(val_df)}   Test: {len(test_df)}")
overlap = set(train_df["text"]) & set(val_df["text"])
print(f"Duplicate utterances shared between train and val: {len(overlap)} (should be 0)")

from transformers import AutoTokenizer
from datasets import Dataset

MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LEN = 64  # justified by the word-count EDA above

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

sample_enc = tokenizer(train_df["clean_text"].iloc[0], truncation=True, max_length=MAX_LEN)
print("Example text  :", train_df["clean_text"].iloc[0])
print("Token ids     :", sample_enc["input_ids"])
print("Tokens        :", tokenizer.convert_ids_to_tokens(sample_enc["input_ids"]))

def to_hf_dataset(df, text_col="clean_text", label_col="label"):
    """Turn a pandas DataFrame into a tokenized torch-formatted HF Dataset.
    The label column is renamed to 'labels' (plural) because that is the
    keyword argument name `AutoModelForSequenceClassification.forward` expects;
    the Trainer only computes a loss when a 'labels' key is present in the batch.
    """
    d = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "labels"})
    ds = Dataset.from_pandas(d, preserve_index=False)
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN),
        batched=True,
    )
    ds = ds.remove_columns(["text"])
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

train_ds = to_hf_dataset(train_df)
val_ds = to_hf_dataset(val_df)
test_ds = to_hf_dataset(test_df)

print(train_ds)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed

set_seed(SEED)

intent_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_INTENT_LABELS,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

INTENT_OUTPUT_DIR = "./intent_model_ckpt"

training_args = TrainingArguments(
    output_dir=INTENT_OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=SEED,
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=intent_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
print(train_result.metrics)

test_results = trainer.evaluate(test_ds)
print("Test set metrics:")
for k, v in test_results.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

preds_output = trainer.predict(test_ds)
y_true_intent = preds_output.label_ids
y_pred_intent = np.argmax(preds_output.predictions, axis=-1)

print("\nFull 77-class classification report:\n")
print(classification_report(y_true_intent, y_pred_intent, target_names=label_names, zero_division=0))

# Confusion matrix restricted to the 15 most frequent intents for readability;
# the full 77-class performance is already captured in the classification_report above.
TOP_N = 15
top_labels_idx = train_df["label"].value_counts().nlargest(TOP_N).index.tolist()

mask = np.isin(y_true_intent, top_labels_idx)
cm = confusion_matrix(y_true_intent[mask], y_pred_intent[mask], labels=top_labels_idx)
cm_labels = [id2label[i] for i in top_labels_idx]

plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=cm_labels, yticklabels=cm_labels, cmap="Blues", annot=True, fmt="d", cbar=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title(f"Confusion Matrix — Top {TOP_N} Most Frequent Intents (Test Set)")
plt.xlabel("Predicted intent")
plt.ylabel("True intent")
plt.tight_layout()
plt.show()

# Persist the fine-tuned intent model for the inference pipeline used later in the notebook.
INTENT_MODEL_DIR = "./intent_model_final"
trainer.save_model(INTENT_MODEL_DIR)
tokenizer.save_pretrained(INTENT_MODEL_DIR)

from transformers import pipeline as hf_pipeline

intent_classifier = hf_pipeline(
    "text-classification",
    model=INTENT_MODEL_DIR,
    tokenizer=INTENT_MODEL_DIR,
    device=PIPE_DEVICE,
    top_k=1,
)

def predict_intent(text: str):
    result = intent_classifier(clean_text(text))
    result = result[0] if isinstance(result[0], dict) else result[0][0]
    return result["label"], float(result["score"])

print(predict_intent("I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking."))

sentiment_pipeline = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=PIPE_DEVICE,
)

print("Model label mapping:", sentiment_pipeline.model.config.id2label)

def predict_sentiment(text: str):
    result = sentiment_pipeline(clean_text(text)[:512])[0]
    return result["label"].capitalize(), float(result["score"])

qualitative_examples = [
    "I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking.",
    "Thank you so much, the refund was processed super quickly!",
    "Can you tell me how to update my mailing address?",
]
for ex in qualitative_examples:
    label, score = predict_sentiment(ex)
    print(f"[{label} ({score:.2f})]  {ex}")

# Small, transparently hand-labeled spot-check set (NOT a formal benchmark — see markdown above).
sentiment_spotcheck = [
    ("I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking.", "Negative"),
    ("This is absolutely unacceptable, I have been charged twice for the same order.", "Negative"),
    ("I am very disappointed with how long this refund is taking.", "Negative"),
    ("Nobody has responded to my emails and I'm getting really frustrated.", "Negative"),
    ("The product arrived broken and customer service has been no help at all.", "Negative"),
    ("I want to cancel my subscription, this service has gone downhill.", "Negative"),
    ("Thank you so much, the refund was processed super quickly!", "Positive"),
    ("Great service as always, really appreciate the quick response.", "Positive"),
    ("I just wanted to say the new update makes everything so much easier, thank you!", "Positive"),
    ("Your support team resolved my issue in minutes, fantastic job.", "Positive"),
    ("I'm really happy with how smoothly the account setup went.", "Positive"),
    ("Thanks for the quick reply, that answers my question perfectly.", "Positive"),
    ("Can you tell me how to update my mailing address?", "Neutral"),
    ("What are your customer service hours?", "Neutral"),
    ("How do I set up a recurring transfer to my savings account?", "Neutral"),
    ("Where can I find my account statements from last year?", "Neutral"),
    ("Is there a way to change the currency displayed in the app?", "Neutral"),
    ("What is the daily withdrawal limit for my account?", "Neutral"),
]

texts, gold_sentiment = zip(*sentiment_spotcheck)
pred_sentiment = [predict_sentiment(t)[0] for t in texts]
sentiment_spotcheck_acc = accuracy_score(gold_sentiment, pred_sentiment)

print(f"Spot-check accuracy: {sentiment_spotcheck_acc:.2%}  (n={len(sentiment_spotcheck)})")
pd.DataFrame({"text": texts, "gold": gold_sentiment, "predicted": pred_sentiment})

HIGH_URGENCY_PATTERNS = [
    r"\bimmediately\b", r"\burgent(ly)?\b", r"\basap\b", r"\bright now\b",
    r"\bemergency\b", r"\bcancel\b", r"\bfraud\b", r"\bstolen\b", r"\bunauthori[sz]ed\b",
    r"\blegal action\b", r"\bcomplaint\b", r"\bnot responding\b", r"\bno response\b",
    r"\bstill waiting\b", r"\bthird time\b", r"\bunacceptable\b", r"\bangry\b",
    r"\bfor \d+ (day|days|week|weeks|hour|hours)\b", r"\bcharged twice\b",
    r"\btoday\b.*\bneed\b", r"!{1,}",
]
MEDIUM_URGENCY_PATTERNS = [
    r"\bwhen will\b", r"\bhow long\b", r"\bplease help\b", r"\bissue\b",
    r"\bproblem\b", r"\bnot working\b", r"\berror\b", r"\bconfused\b",
    r"\bcan someone\b", r"\bwhy (was|is|did)\b",
]

def urgency_heuristic_score(text: str) -> int:
    t = text.lower()
    score = 0
    for pat in HIGH_URGENCY_PATTERNS:
        if re.search(pat, t):
            score += 2
    for pat in MEDIUM_URGENCY_PATTERNS:
        if re.search(pat, t):
            score += 1
    caps_words = re.findall(r"\b[A-Z]{3,}\b", text)  # shouting signal, checked on ORIGINAL casing
    score += len(caps_words)
    return score

def urgency_weak_label(text: str) -> str:
    score = urgency_heuristic_score(text)
    if score >= 3:
        return "High"
    elif score >= 1:
        return "Medium"
    else:
        return "Low"

URGENCY_LABELS = ["Low", "Medium", "High"]
urgency_id2label = {i: l for i, l in enumerate(URGENCY_LABELS)}
urgency_label2id = {l: i for i, l in enumerate(URGENCY_LABELS)}

for df_ in (train_df, val_df, test_df):
    df_["urgency_weak"] = df_["clean_text"].apply(urgency_weak_label)
    df_["urgency_label_id"] = df_["urgency_weak"].map(urgency_label2id)

print("Weak-label distribution (train):")
print(train_df["urgency_weak"].value_counts())

class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_df["urgency_label_id"].values,
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print("Class weights (Low, Medium, High):", class_weights_np)

urgency_train_ds = to_hf_dataset(train_df, label_col="urgency_label_id")
urgency_val_ds = to_hf_dataset(val_df, label_col="urgency_label_id")
urgency_test_ds = to_hf_dataset(test_df, label_col="urgency_label_id")

from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    """Trainer variant that applies class weights to the cross-entropy loss,
    needed because the weak urgency labels are imbalanced (see above)."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

urgency_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(URGENCY_LABELS),
    id2label=urgency_id2label,
    label2id=urgency_label2id,
)

URGENCY_OUTPUT_DIR = "./urgency_model_ckpt"

urgency_training_args = TrainingArguments(
    output_dir=URGENCY_OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=SEED,
    logging_steps=50,
    report_to="none",
)

urgency_trainer = WeightedTrainer(
    class_weights=class_weights,
    model=urgency_model,
    args=urgency_training_args,
    train_dataset=urgency_train_ds,
    eval_dataset=urgency_val_ds,
    compute_metrics=compute_metrics,
)

urgency_train_result = urgency_trainer.train()
print(urgency_train_result.metrics)

# (a) Evaluation against the heuristic-labeled test split.
urgency_test_results = urgency_trainer.evaluate(urgency_test_ds)
urgency_weak_test_f1 = urgency_test_results["eval_f1"]
print("Urgency model vs. WEAK (heuristic) test labels:")
for k, v in urgency_test_results.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

urgency_preds_output = urgency_trainer.predict(urgency_test_ds)
y_true_urg = urgency_preds_output.label_ids
y_pred_urg = np.argmax(urgency_preds_output.predictions, axis=-1)
print("\n" + classification_report(y_true_urg, y_pred_urg, target_names=URGENCY_LABELS, zero_division=0))

cm_urg = confusion_matrix(y_true_urg, y_pred_urg, labels=[0, 1, 2])
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm_urg, annot=True, fmt="d", cmap="Oranges", xticklabels=URGENCY_LABELS, yticklabels=URGENCY_LABELS)
plt.title("Urgency Confusion Matrix (vs. heuristic test labels)")
plt.xlabel("Predicted")
plt.ylabel("Weak label")
plt.tight_layout()
plt.show()

# Save the urgency model and build an inference helper before the gold-set check.
URGENCY_MODEL_DIR = "./urgency_model_final"
urgency_trainer.save_model(URGENCY_MODEL_DIR)
tokenizer.save_pretrained(URGENCY_MODEL_DIR)

urgency_classifier = hf_pipeline(
    "text-classification",
    model=URGENCY_MODEL_DIR,
    tokenizer=URGENCY_MODEL_DIR,
    device=PIPE_DEVICE,
    top_k=1,
)

def predict_urgency(text: str):
    result = urgency_classifier(clean_text(text))
    result = result[0] if isinstance(result[0], dict) else result[0][0]
    return result["label"], float(result["score"])

# (b) Evaluation against a small, hand-written, human-judged GOLD set.
# These 24 examples were written and urgency-labeled by hand for this notebook
# specifically as a real-world sanity check -- they are NOT drawn from Banking77
# and were not used anywhere in training, so this measures genuine generalization
# beyond the heuristic's exact keyword list.
urgency_gold_set = [
    ("I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking.", "High"),
    ("This is the third time I'm emailing about the same fraud charge on my card and no one has helped me!", "High"),
    ("My card was stolen and someone just used it to make a purchase, please block it immediately.", "High"),
    ("I am extremely angry, I was charged twice for the same transaction and I need this fixed right now.", "High"),
    ("My account has been locked and I have a flight in two hours, I need access immediately.", "High"),
    ("I have already called customer service five times about this refund and nobody calls me back, this is unacceptable.", "High"),
    ("Please cancel my subscription immediately, I was never told about this renewal charge.", "High"),
    ("I need this resolved today, I have already waited a week and I'm losing money because of this.", "High"),
    ("Hi, I noticed an extra charge on my statement, can someone explain what it's for?", "Medium"),
    ("My app keeps showing an error when I try to transfer money, could you help me fix this?", "Medium"),
    ("When will my new card arrive? It's been a few days since I ordered it.", "Medium"),
    ("I'm a bit confused about how the loyalty points work, can you clarify?", "Medium"),
    ("There seems to be a problem with my direct debit setup, can someone look into it?", "Medium"),
    ("I tried resetting my password but it's not working, any idea why?", "Medium"),
    ("Can you tell me why my last payment failed?", "Medium"),
    ("I'd like to know how long the verification process usually takes.", "Medium"),
    ("Can you tell me how to update my mailing address?", "Low"),
    ("What are your customer service hours?", "Low"),
    ("Just wanted to say thanks, the new app update looks great!", "Low"),
    ("How do I set up a recurring transfer to my savings account?", "Low"),
    ("Where can I find my account statements from last year?", "Low"),
    ("Is there a way to change the currency displayed in the app?", "Low"),
    ("Can I add a second card to my account?", "Low"),
    ("What is the daily withdrawal limit for my account?", "Low"),
]

gold_texts, gold_urgency = zip(*urgency_gold_set)
pred_urgency = [predict_urgency(t)[0] for t in gold_texts]
urgency_gold_acc = accuracy_score(gold_urgency, pred_urgency)
urgency_gold_f1 = f1_score(gold_urgency, pred_urgency, average="weighted", zero_division=0)

print(f"Gold spot-check accuracy: {urgency_gold_acc:.2%}   weighted F1: {urgency_gold_f1:.3f}   (n={len(urgency_gold_set)})")
gold_result_df = pd.DataFrame({"text": gold_texts, "gold": gold_urgency, "predicted": pred_urgency})
gold_result_df

from keybert import KeyBERT

ner_pipeline = hf_pipeline(
    "ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=PIPE_DEVICE
)
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

MONEY_RE = re.compile(r"\$\s?\d+(?:\.\d{1,2})?|\b\d+(?:\.\d{1,2})?\s?(?:usd|dollars|eur|euros|gbp|pounds)\b", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d+\s+(?:day|days|week|weeks|month|months|hour|hours)\b|yesterday|today|tomorrow)\b", re.I)

def extract_entities_and_topics(text: str, top_k_topics: int = 5):
    ner_ents = ner_pipeline(text)
    named_entities = [
        {"text": e["word"], "type": e["entity_group"], "score": round(float(e["score"]), 3)}
        for e in ner_ents
    ]
    money_matches = [m.group() for m in MONEY_RE.finditer(text)]
    date_matches = [m.group() for m in DATE_RE.finditer(text)]

    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_k_topics
    )
    topics = [kw for kw, _score in keywords]

    return {
        "named_entities": named_entities,
        "money_mentions": money_matches,
        "date_mentions": date_matches,
        "topics": topics,
    }

demo_text = "I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking."
extract_entities_and_topics(demo_text)

summarizer = hf_pipeline(
    "summarization", model="philschmid/distilbart-cnn-12-6-samsum", device=PIPE_DEVICE
)

def format_for_dialogue_summarizer(text: str) -> str:
    return f"Customer: {text.strip()}"

def summarize_conversation(text: str, max_length: int = 40, min_length: int = 8) -> str:
    word_count = len(text.split())
    if word_count < 6:
        return text.strip()  # too short to meaningfully compress further
    formatted = format_for_dialogue_summarizer(text)
    max_len = max(min_length + 4, min(max_length, word_count + 5))
    result = summarizer(formatted, max_length=max_len, min_length=min_length, do_sample=False)
    return result[0]["summary_text"].strip()

for ex in [
    "I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking.",
    "My card was stolen and someone just used it to make a purchase, please block it immediately and send me a replacement.",
]:
    print(f"INPUT  : {ex}")
    print(f"SUMMARY: {summarize_conversation(ex)}\n")

def analyze_conversation(text: str) -> dict:
    cleaned = clean_text(text)

    intent, intent_conf = predict_intent(cleaned)
    sentiment, sentiment_conf = predict_sentiment(cleaned)
    urgency, urgency_conf = predict_urgency(cleaned)
    ents_topics = extract_entities_and_topics(cleaned)
    summary = summarize_conversation(cleaned)

    return {
        "input_text": text,
        "intent": {"label": intent, "confidence": round(intent_conf, 3)},
        "sentiment": {"label": sentiment, "confidence": round(sentiment_conf, 3)},
        "urgency": {"label": urgency, "confidence": round(urgency_conf, 3)},
        "entities": ents_topics["named_entities"],
        "topics": ents_topics["topics"],
        "money_mentions": ents_topics["money_mentions"],
        "date_mentions": ents_topics["date_mentions"],
        "summary": summary,
    }

def pretty_print_analysis(result: dict) -> None:
    print("=" * 78)
    print(f"INPUT: {result['input_text']}")
    print("-" * 78)
    print(f"Intent      : {result['intent']['label']}  (confidence: {result['intent']['confidence']})")
    print(f"Sentiment   : {result['sentiment']['label']}  (confidence: {result['sentiment']['confidence']})")
    print(f"Urgency     : {result['urgency']['label']}  (confidence: {result['urgency']['confidence']})")
    print(f"Entities    : {result['entities']}")
    print(f"Topics      : {result['topics']}")
    print(f"Money       : {result['money_mentions']}")
    print(f"Dates/Times : {result['date_mentions']}")
    print(f"Summary     : {result['summary']}")
    print("=" * 78)

demo_conversations = [
    "I've been trying to contact the owner for three days but nobody is responding. I want to cancel my booking.",
    "Hi! Just wanted to say thank you, my card arrived earlier than expected and everything works perfectly.",
    "My card was stolen and someone just used it to make a $250 purchase, please block it immediately!!!",
    "Hey, quick question - how do I change the currency shown in the app? Not urgent at all.",
    "I was charged twice for the same transaction on 03/14/2026 and nobody has responded to my last two emails about it.",
    "Can you explain how the loyalty points program works? I signed up last month and I'm a little confused.",
]

for convo in demo_conversations:
    result = analyze_conversation(convo)
    pretty_print_analysis(result)

# Intent misclassifications on the test set
misclassified_mask = y_true_intent != y_pred_intent
mis_df = test_df.iloc[np.where(misclassified_mask)[0]].copy()
mis_df["true_intent"] = [id2label[i] for i in y_true_intent[misclassified_mask]]
mis_df["pred_intent"] = [id2label[i] for i in y_pred_intent[misclassified_mask]]

print(f"Total misclassified: {misclassified_mask.sum()} / {len(y_true_intent)} "
      f"({misclassified_mask.mean():.2%})")
mis_df[["text", "true_intent", "pred_intent"]].sample(min(10, len(mis_df)), random_state=SEED)

# Which intent pairs are confused most often?
confusion_pairs = (
    mis_df.groupby(["true_intent", "pred_intent"]).size().reset_index(name="count")
    .sort_values("count", ascending=False).head(10)
)
confusion_pairs

print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"Intent Classification — DistilBERT fine-tuned, {NUM_INTENT_LABELS}-class (Banking77 test set)")
print(f"   Accuracy          : {test_results['eval_accuracy']:.4f}")
print(f"   Weighted F1       : {test_results['eval_f1']:.4f}")
print(f"   Weighted Precision: {test_results['eval_precision']:.4f}")
print(f"   Weighted Recall   : {test_results['eval_recall']:.4f}")
print()
print("Urgency Detection — DistilBERT fine-tuned on weak-supervised labels")
print(f"   Weak-label test F1 (vs. heuristic)  : {urgency_weak_test_f1:.4f}")
print(f"   Gold spot-check accuracy (n={len(urgency_gold_set)}): {urgency_gold_acc:.4f}")
print(f"   Gold spot-check weighted F1         : {urgency_gold_f1:.4f}")
print()
print(f"Sentiment — pretrained RoBERTa (zero-shot on this domain)")
print(f"   Spot-check accuracy (n={len(sentiment_spotcheck)}): {sentiment_spotcheck_acc:.4f}")
print("=" * 60)

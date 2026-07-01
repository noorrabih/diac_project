"""
Fine-tuning NeoAraBERT on the BAREC dataset for Arabic readability classification.

NeoAraBERT is built on the NeoBERT architecture (custom model code),
so we use trust_remote_code=True and AutoModelForSequenceClassification.

Dataset: CAMeL-Lab/BAREC-Corpus-v1.0
Task: 19-class readability level classification (Readability_Level_19, labels shifted to 0-indexed)

Usage:
    pip install transformers datasets torch scikit-learn
    python finetune_neoarabert_barec.py

NOTE: Replace MODEL_NAME below with the actual HuggingFace ID when the model is public,
      e.g. "NeoAraBERT/NeoAraBERT" or the org/repo slug from the model card.
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    mean_absolute_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME   = "U4RASD/NeoAraBERT"   # ← update if the HF model slug differs
INPUT_VAR    = "Word"                 # dataset column to use as input text
                                          # alternatives: "Paragraph", "Document"
NUM_LABELS   = 19                         # Readability_Level_19 has levels 1-19
MAX_LENGTH   = 512                        # NeoAraBERT supports up to 4096; use 512 for speed
BATCH_SIZE   = 16
GRAD_ACCUM   = 2                          # effective batch = 16 * 2 = 32
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
OUTPUT_DIR   = "neoarabert-barec-finetuned-word"
SEED         = 42

# Coarser-grained level mappings (19-level Readability_Level_19 -> 7/5/3-level)
barec_7_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4, 12: 5, 13: 5, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 7
}
barec_5_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5
}
barec_3_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3
}

dev_out_xlsx = f"{OUTPUT_DIR}/dev_predictions.xlsx"
test_out_xlsx = f"{OUTPUT_DIR}/test_predictions.xlsx"

# ─── Load Dataset ─────────────────────────────────────────────────────────────

print("Loading BAREC dataset …")
dataset = load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0")

# Labels are 1-indexed in the dataset; shift to 0-indexed
dataset_train = {
    "text":  list(dataset["train"][INPUT_VAR]),
    "label": [l - 1 for l in dataset["train"]["Readability_Level_19"]],
}
dataset_dev = {
    "text":  list(dataset["dev"][INPUT_VAR]),
    "label": [l - 1 for l in dataset["dev"]["Readability_Level_19"]],
}
dataset_test = {
    "text":  list(dataset["test"][INPUT_VAR]),
    "label": [l - 1 for l in dataset["test"]["Readability_Level_19"]],
}

print(f"Train: {len(dataset_train['text'])} | Dev: {len(dataset_dev['text'])} | Test: {len(dataset_test['text'])}")

# ─── Tokenizer ────────────────────────────────────────────────────────────────

print(f"Loading tokenizer from '{MODEL_NAME}' …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class BARECDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = BARECDataset(dataset_train["text"], dataset_train["label"], tokenizer, MAX_LENGTH)
dev_dataset   = BARECDataset(dataset_dev["text"],   dataset_dev["label"],   tokenizer, MAX_LENGTH)
test_dataset  = BARECDataset(dataset_test["text"],  dataset_test["label"],  tokenizer, MAX_LENGTH)

# ─── Model ────────────────────────────────────────────────────────────────────

print(f"Loading model from '{MODEL_NAME}' …")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    trust_remote_code=True,   # required for NeoAraBERT's custom architecture
    ignore_mismatched_sizes=True,  # safe when adding a new classification head
)

# ─── Metrics ──────────────────────────────────────────────────────────────────

def accuracy_with_margin_classification(y_true, y_pred, margin=1):
    """
    Calculates accuracy with a specified margin of error for classification.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        margin (int): Acceptable difference between true and predicted classes.

    Returns:
        float: Accuracy with margin.
    """
    correct_predictions = np.sum(np.abs(np.array(y_true) - np.array(y_pred)) <= margin)
    return correct_predictions / len(y_true)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    assert len(preds) == len(p.label_ids)
    print(classification_report(p.label_ids, preds, digits=4))
    print(confusion_matrix(p.label_ids, preds))
    preds_19 = list(preds)
    labels_19 = list(p.label_ids)
    preds_7 = [barec_7_dict[i + 1] for i in preds_19]
    labels_7 = [barec_7_dict[i + 1] for i in labels_19]
    preds_5 = [barec_5_dict[i + 1] for i in preds_19]
    labels_5 = [barec_5_dict[i + 1] for i in labels_19]
    preds_3 = [barec_3_dict[i + 1] for i in preds_19]
    labels_3 = [barec_3_dict[i + 1] for i in labels_19]

    macro_f1 = f1_score(p.label_ids, preds, average='macro')
    macro_precision = precision_score(p.label_ids, preds, average='macro')
    macro_recall = recall_score(p.label_ids, preds, average='macro')
    acc = accuracy_score(p.label_ids, preds)
    acc_with_margin = accuracy_with_margin_classification(p.label_ids, preds, margin=1)
    acc_7 = accuracy_score(labels_7, preds_7)
    acc_5 = accuracy_score(labels_5, preds_5)
    acc_3 = accuracy_score(labels_3, preds_3)
    QWK = cohen_kappa_score(p.label_ids, preds, weights='quadratic')
    dist = mean_absolute_error(p.label_ids, preds)
    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'accuracy': acc,
        'accuracy_with_margin': acc_with_margin,
        'Distance': dist,
        'Quadratic Weighted Kappa': QWK,
        'accuracy_7': acc_7,
        'accuracy_5': acc_5,
        'accuracy_3': acc_3,
    }

# ─── Training Arguments ───────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # gradient_accumulation_steps=GRAD_ACCUM,
    # learning_rate=LR,
    # weight_decay=WEIGHT_DECAY,
    # warmup_ratio=0.1,
    # lr_scheduler_type="linear",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),   # enable mixed precision if GPU available
    seed=SEED,
    report_to="none",                 # set to "wandb" or "tensorboard" if desired
)

# ─── Trainer ──────────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─── Train ────────────────────────────────────────────────────────────────────

print("\n=== Starting fine-tuning ===")
trainer.train()

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def rank_simple(vector):
    return [i + 1 for i in sorted(range(len(vector)), key=vector.__getitem__, reverse=True)]


def save_predictions(preds, texts_list, labels_list, out_xlsx):
    probs = {i + 1: [] for i in range(NUM_LABELS)}
    ranks = {i + 1: [] for i in range(NUM_LABELS)}
    texts = []
    labels = []
    predictions = []
    labels_7 = []
    predictions_7 = []
    labels_5 = []
    predictions_5 = []
    labels_3 = []
    predictions_3 = []
    is_equal = []
    is_equal_7 = []
    is_equal_5 = []
    is_equal_3 = []
    within_one = []
    within_top_2 = []
    within_top_3 = []
    within_top_4 = []
    rank_of_correct = []
    diff = []

    for i in range(len(preds)):
        texts.append(texts_list[i])
        labels.append(labels_list[i] + 1)
        predictions.append(argmax(preds[i]) + 1)
        labels_7.append(barec_7_dict[labels[-1]])
        predictions_7.append(barec_7_dict[predictions[-1]])
        labels_5.append(barec_5_dict[labels[-1]])
        predictions_5.append(barec_5_dict[predictions[-1]])
        labels_3.append(barec_3_dict[labels[-1]])
        predictions_3.append(barec_3_dict[predictions[-1]])
        softs = softmax(preds[i])
        rank = rank_simple(softs)

        is_equal.append(1 if labels[-1] == predictions[-1] else 0)
        is_equal_7.append(1 if labels_7[-1] == predictions_7[-1] else 0)
        is_equal_5.append(1 if labels_5[-1] == predictions_5[-1] else 0)
        is_equal_3.append(1 if labels_3[-1] == predictions_3[-1] else 0)
        within_one.append(1 if abs(labels[-1] - predictions[-1]) <= 1 else 0)
        within_top_2.append(1 if labels[-1] in rank[:2] else 0)
        within_top_3.append(1 if labels[-1] in rank[:3] else 0)
        within_top_4.append(1 if labels[-1] in rank[:4] else 0)

        rank_of_correct.append(rank.index(labels[-1]) + 1)
        diff.append(max(labels[-1], predictions[-1]) - min(labels[-1], predictions[-1]))
        for j in range(NUM_LABELS):
            probs[j + 1].append(softs[j])
            ranks[j + 1].append(rank[j])

    QWK = cohen_kappa_score(labels, predictions, weights='quadratic')
    acc = sum(is_equal) / len(is_equal)
    acc_5 = sum(is_equal_5) / len(is_equal_5)
    acc_3 = sum(is_equal_3) / len(is_equal_3)
    acc_within_one_level = sum(within_one) / len(within_one)
    avg_distance = sum(diff) / len(diff)

    print(f"Accuracy: {acc * 100:.4f}")
    print(f"Accuracy with margin of one level: {acc_within_one_level * 100:.4f}")
    print(f"Average distance between labels and predictions: {avg_distance:.6f}")
    print(f"Quadratic Weighted Kappa: {QWK * 100:.4f}")
    print(f"Accuracy_5: {acc_5 * 100:.4f}")
    print(f"Accuracy_3: {acc_3 * 100:.4f}")

    v = {
        'text': texts,
        'label': labels,
        'prediction': predictions,
        'is_equal': is_equal,
        'within_one_level': within_one,
        'within_top2_ranks': within_top_2,
        'within_top3_ranks': within_top_3,
        'within_top4_ranks': within_top_4,
        'rank_of_correct_label': rank_of_correct,
        'diff': diff,
    }
    for i in range(1, NUM_LABELS + 1):
        v['p' + str(i)] = probs[i]
    for i in range(1, NUM_LABELS + 1):
        v['rank' + str(i)] = ranks[i]

    final_df = pd.DataFrame.from_dict(v)
    final_df.to_excel(out_xlsx, index=False)


# ─── Evaluate on Dev ──────────────────────────────────────────────────────────

print("\n=== Dev set evaluation ===")
preds, labels, metrics = trainer.predict(dev_dataset, metric_key_prefix="eval")
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

save_predictions(preds, dataset_dev["text"], dataset_dev["label"], dev_out_xlsx)


# ─── Evaluate on Test ─────────────────────────────────────────────────────────

print("\n=== Test set evaluation ===")
preds, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

save_predictions(preds, dataset_test["text"], dataset_test["label"], test_out_xlsx)


# ─── Save ─────────────────────────────────────────────────────────────────────

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel and tokenizer saved to '{OUTPUT_DIR}'")
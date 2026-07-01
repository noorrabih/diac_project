# Regression (MSE) variant of finetune_neoarabert_barec.py
#
# Key differences from the classification version:
#   - model head outputs a single scalar (num_labels=1)
#   - loss = MSE(predicted_scalar, true_label)
#   - at inference: prediction = round(output) clipped to [0, n_levels-1]
#   - no class-probability or rank columns in output xlsx


from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, mean_absolute_error, cohen_kappa_score
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train model with MSE regression")
parser.add_argument('--model', type=str, required=True, help='Pre-trained model checkpoint')
parser.add_argument('--input_var', type=str, required=True, help='Input variant (e.g., Word, D3Tok)')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the trained model')
parser.add_argument('--output_path', type=str, required=True, help='Directory to save the output xlsx files')
parser.add_argument('--max_length', type=int, default=512, help='Max tokenized sequence length')
args = parser.parse_args()

loss_type = "Reg"
checkpoint = args.model
input_var = args.input_var
save_dir_base = args.save_dir
output_path = args.output_path
max_length = args.max_length

barec_7_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4, 12: 5, 13: 5, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 7
}
barec_5_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5
}
barec_3_dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3
}

save_dir = os.path.join(save_dir_base, f"{checkpoint.split('/')[-1]}_{input_var}_{loss_type}_19levels")

dev_out_xlsx = os.path.join(output_path, f"dev_{checkpoint.split('/')[-1]}_{input_var}_{loss_type}_19levels.xlsx")
test_out_xlsx = os.path.join(output_path, f"test_{checkpoint.split('/')[-1]}_{input_var}_{loss_type}_19levels.xlsx")

print(f"loss: {loss_type}, model: {checkpoint.split('/')[-1]}, input_var: {input_var}, levels: 19")

n_levels = 19


class RegTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = F.mse_loss(logits, labels)
        inputs["labels"] = labels.long()
        return (loss, outputs) if return_outputs else loss


set_seed(42)

dataset = load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0")

dataset_train = {
    'text': dataset['train'][input_var],
    'label': [l-1 for l in dataset['train']['Readability_Level_19']]
}
dataset_dev = {
    'text': dataset['dev'][input_var],
    'label': [l-1 for l in dataset['dev']['Readability_Level_19']]
}
dataset_test = {
    'text': dataset['test'][input_var],
    'label': [l-1 for l in dataset['test']['Readability_Level_19']]
}

dataset_train = datasets.Dataset.from_dict(dataset_train)
dataset_dev = datasets.Dataset.from_dict(dataset_dev)
dataset_test = datasets.Dataset.from_dict(dataset_test)

# dataset_train = dataset_train.filter(lambda x: x['label'] < 11)
# dataset_dev = dataset_dev.filter(lambda x: x['label'] < 11)
# dataset_test = dataset_test.filter(lambda x: x['label'] < 11)

dataset['train'] = dataset_train
dataset['dev'] = dataset_dev
dataset['test'] = dataset_test

print(dataset_train.column_names)
print(dataset_dev.column_names)
print(dataset_test.column_names)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=1, ignore_mismatched_sizes=True, trust_remote_code=True
    )
    for param in model.parameters():
        param.data = param.data.contiguous()
    return model


def accuracy_with_margin_classification(y_true, y_pred, margin=1):
    correct_predictions = np.sum(np.abs(np.array(y_true) - np.array(y_pred)) <= margin)
    return correct_predictions / len(y_true)

def compute_metrics(p):
    preds = np.clip(np.round(p.predictions.squeeze()), 0, n_levels - 1).astype(int)
    labels_arr = p.label_ids
    assert len(preds) == len(labels_arr)
    print(classification_report(labels_arr, preds, digits=4))
    print(confusion_matrix(labels_arr, preds))
    preds_19 = list(preds)
    labels_19 = list(labels_arr)
    preds_7 = [barec_7_dict[i+1] for i in preds_19]
    labels_7 = [barec_7_dict[i+1] for i in labels_19]
    preds_5 = [barec_5_dict[i+1] for i in preds_19]
    labels_5 = [barec_5_dict[i+1] for i in labels_19]
    preds_3 = [barec_3_dict[i+1] for i in preds_19]
    labels_3 = [barec_3_dict[i+1] for i in labels_19]

    macro_f1 = f1_score(labels_arr, preds, average='macro')
    macro_precision = precision_score(labels_arr, preds, average='macro')
    macro_recall = recall_score(labels_arr, preds, average='macro')
    acc = accuracy_score(labels_arr, preds)
    acc_with_margin = accuracy_with_margin_classification(labels_arr, preds, margin=1)
    acc_7 = accuracy_score(labels_7, preds_7)
    acc_5 = accuracy_score(labels_5, preds_5)
    acc_3 = accuracy_score(labels_3, preds_3)
    QWK = cohen_kappa_score(labels_arr, preds, weights='quadratic')
    dist = mean_absolute_error(labels_arr, preds)
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
        'accuracy_3': acc_3
    }


training_args = TrainingArguments(save_dir,
                                  eval_strategy="epoch",
                                  num_train_epochs=6,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_loss",
                                  greater_is_better=False,
                                  save_strategy="epoch",
                                  save_total_limit=1,
                                  )


trainer = RegTrainer(model_init=model_init,
                     args=training_args,
                     train_dataset=tokenized_datasets['train'],
                     eval_dataset=tokenized_datasets['dev'],
                     data_collator=data_collator,
                     tokenizer=tokenizer,
                     compute_metrics=compute_metrics)


trainer.train()


def reg_predict(raw_pred):
    return int(np.clip(round(float(raw_pred[0])), 0, n_levels - 1)) + 1


preds, labels, metrics = trainer.predict(tokenized_datasets['dev'], metric_key_prefix="eval")
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


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
diff = []

for i in range(len(preds)):
    texts.append(list(dataset_dev['text'])[i])
    labels.append(list(dataset_dev['label'])[i]+1)
    predictions.append(reg_predict(preds[i]))
    labels_7.append(barec_7_dict[labels[-1]])
    predictions_7.append(barec_7_dict[predictions[-1]])
    labels_5.append(barec_5_dict[labels[-1]])
    predictions_5.append(barec_5_dict[predictions[-1]])
    labels_3.append(barec_3_dict[labels[-1]])
    predictions_3.append(barec_3_dict[predictions[-1]])

    is_equal.append(1 if labels[-1] == predictions[-1] else 0)
    is_equal_7.append(1 if labels_7[-1] == predictions_7[-1] else 0)
    is_equal_5.append(1 if labels_5[-1] == predictions_5[-1] else 0)
    is_equal_3.append(1 if labels_3[-1] == predictions_3[-1] else 0)
    within_one.append(1 if abs(labels[-1] - predictions[-1]) <= 1 else 0)
    diff.append(abs(labels[-1] - predictions[-1]))


QWK = cohen_kappa_score(labels, predictions, weights='quadratic')
acc = sum(is_equal)/len(is_equal)
acc_5 = sum(is_equal_5)/len(is_equal_5)
acc_3 = sum(is_equal_3)/len(is_equal_3)
acc_within_one_level = sum(within_one)/len(within_one)
avg_distance = sum(diff)/len(diff)

print(f"Accuracy: {acc*100:.4f}")
print(f"Accuracy with margin of one level: {acc_within_one_level*100:.4f}")
print(f"Average distance between labels and predictions: {avg_distance:.6f}")
print(f"Quadratic Weighted Kappa: {QWK*100:.4f}")
print(f"Accuracy_5: {acc_5*100:.4f}")
print(f"Accuracy_3: {acc_3*100:.4f}")


v = {
    'text': texts,
    'label': labels,
    'prediction': predictions,
    'is_equal': is_equal,
    'within_one_level': within_one,
    'diff': diff
}

final_df = pd.DataFrame.from_dict(v)
final_df.to_excel(dev_out_xlsx, index=False)


############Test#################

preds, labels, metrics = trainer.predict(tokenized_datasets['test'], metric_key_prefix="test")
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

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
diff = []

for i in range(len(preds)):
    texts.append(list(dataset_test['text'])[i])
    labels.append(list(dataset_test['label'])[i]+1)
    predictions.append(reg_predict(preds[i]))
    labels_7.append(barec_7_dict[labels[-1]])
    predictions_7.append(barec_7_dict[predictions[-1]])
    labels_5.append(barec_5_dict[labels[-1]])
    predictions_5.append(barec_5_dict[predictions[-1]])
    labels_3.append(barec_3_dict[labels[-1]])
    predictions_3.append(barec_3_dict[predictions[-1]])

    is_equal.append(1 if labels[-1] == predictions[-1] else 0)
    is_equal_7.append(1 if labels_7[-1] == predictions_7[-1] else 0)
    is_equal_5.append(1 if labels_5[-1] == predictions_5[-1] else 0)
    is_equal_3.append(1 if labels_3[-1] == predictions_3[-1] else 0)
    within_one.append(1 if abs(labels[-1] - predictions[-1]) <= 1 else 0)
    diff.append(abs(labels[-1] - predictions[-1]))


QWK = cohen_kappa_score(labels, predictions, weights='quadratic')
acc = sum(is_equal)/len(is_equal)
acc_5 = sum(is_equal_5)/len(is_equal_5)
acc_3 = sum(is_equal_3)/len(is_equal_3)
acc_within_one_level = sum(within_one)/len(within_one)
avg_distance = sum(diff)/len(diff)

print(f"Accuracy: {acc*100:.4f}")
print(f"Accuracy with margin of one level: {acc_within_one_level*100:.4f}")
print(f"Average distance between labels and predictions: {avg_distance:.6f}")
print(f"Quadratic Weighted Kappa: {QWK*100:.4f}")
print(f"Accuracy_5: {acc_5*100:.4f}")
print(f"Accuracy_3: {acc_3*100:.4f}")


v = {
    'text': texts,
    'label': labels,
    'prediction': predictions,
    'is_equal': is_equal,
    'within_one_level': within_one,
    'diff': diff
}

final_df = pd.DataFrame.from_dict(v)
final_df.to_excel(test_out_xlsx, index=False)


trainer.save_model(save_dir)

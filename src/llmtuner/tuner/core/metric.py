import numpy as np
from typing import Dict, Sequence, Tuple, Union
import torch.nn.functional as F
import torch
from llmtuner.extras.constants import IGNORE_INDEX
import re

def f1_score(preds, labels):
    f1 = []
    for pred, label in zip(preds, labels):
        f1.append(len(np.intersect1d(pred, label))/len(pred))
    return np.mean(f1)

def compute_nll(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, labels = eval_preds
    f1 = f1_score(preds, labels)
    return {"eval_f1": f1}

def check_eq(left_str, right_str):
    left_matches = re.match(r'(\d+)([+\-*/])(\d+)', left_str)
    if left_matches:
        return eval(left_str) == float(right_str)
    else:
        return False

def compute_rm_acc(eval_preds):
    # (N, s) (N, s+1)
    preds, labels = eval_preds
    score_dict = {}
    score_dict.setdefault(f"acc-err", [])
    score_dict.setdefault(f"acc-cor", [])
    for pred, label in zip(preds, labels):
        t, label = label[0], label[1:]
        # for c, _t in zip(correct, t):
        l = label[label!=-100]
        p = pred[label!=-100] 
        res = l&p 
        score_dict.setdefault(f"acc-{t}-cor", [])
        score_dict.setdefault(f"acc-{t}-err", [])
        score_dict[f"acc-{t}-cor"].extend(res[l==1])
        score_dict[f"acc-{t}-err"].extend(res[l==0])
        score_dict[f"acc-cor"].extend(res[l==1])
        score_dict[f"acc-err"].extend(res[l==0])
    return {k: float(np.mean(v)) for k, v in score_dict.items()}

def compute_acc(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]], tokenizer, data_name) -> Dict[str, float]:
    preds, labels = eval_preds
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    total = 0
    correct = 0
    chunk_size = 4096

    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        label_chunk = torch.tensor(labels[start:end])
        ignore_mask = label_chunk == IGNORE_INDEX
        label_chunk.masked_fill_(ignore_mask, tokenizer.pad_token_id)

        pred_chunk = torch.tensor(preds[start:end])
        if pred_chunk.shape == label_chunk.shape:
            # supppose pred and label have same shape (training stage)
            pred_chunk.masked_fill_(ignore_mask, tokenizer.pad_token_id)
        else:
            pred_chunk.masked_fill_(pred_chunk == IGNORE_INDEX, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(pred_chunk.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(label_chunk.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            if 'cd' in data_name: ## countdown
                subequations = pred.split(',')  # sub-equations
                match = True
                for subeq in subequations:
                    try:
                        left, right = subeq.split('=')
                        match &= check_eq(left, right)
                    except Exception:
                        match = False
                    if not match:
                        break
                answer = label.split('=')[-1]
                pred_ans = pred.split('=')[-1]
                is_correct = match and (answer == pred_ans)
            # elif 'sat' in data_name:
            #     # score_dict["acc"].append(0)

            #     # sat-v2
            #     subphases = pred.split('/')
            #     corr = True
            #     for subphase in subphases:
            #         if 'T' not in subphase:
            #             score_dict["acc"].append(0)
            #             corr = False
            #             break
            #     if corr:
            #         score_dict["acc"].append(1)

            elif 'path' in data_name:
                def reverse_check(gold, pred):
                    try:
                        items = pred.split('/')
                        reversed_pred = "/".join([f'{i.split(",")[1]},{i.split(",")[0]}' for i in items[::-1]])
                        return reversed_pred == gold
                    except Exception:
                        return False

                is_correct = (pred == label) or reverse_check(label, pred)
            else: ## chess, sudoku, prime
                pred_items = pred.strip().split(' ') # pred can have multiple actions
                label_items = label.strip().split(' ') # labels can have multiple actions
                pred_items = pred_items[:len(label_items)] # chess only take next move

                is_correct = pred_items == label_items

            total += 1
            correct += int(is_correct)

    return {"acc": float(correct / total) if total else 0.0}

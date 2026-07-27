#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import DataCollatorForLanguageModeling, Seq2SeqTrainingArguments

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.hparams import DataArguments, DiffusionArguments, FinetuningArguments, ModelArguments
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.mdm.trainer import CustomDiffusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-safe MDM evaluation for large Sudoku splits.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset", default="sudoku_hard")
    parser.add_argument("--dataset_dir", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--model_name_or_path", default="model_config_tiny")
    parser.add_argument("--cutoff_len", type=int, default=164)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--decoding_strategy", default="margin-linear")
    parser.add_argument("--topk_decoding", action="store_true", default=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1024)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def batch_correct(pred_tensor, label_tensor, tokenizer):
    preds = tokenizer.batch_decode(
        pred_tensor.detach().cpu().numpy().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    labels = tokenizer.batch_decode(
        label_tensor.detach().cpu().numpy().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    correct = 0
    for pred, label in zip(preds, labels):
        pred_items = pred.strip().split(" ")
        label_items = label.strip().split(" ")
        pred_items = pred_items[:len(label_items)]
        correct += int(pred_items == label_items)
    return correct, preds, labels


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        checkpoint_dir=[args.checkpoint_dir],
    )
    diff_args = DiffusionArguments(
        diffusion_steps=args.diffusion_steps,
        decoding_strategy=args.decoding_strategy,
        topk_decoding=args.topk_decoding,
    )
    data_args = DataArguments(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        cutoff_len=args.cutoff_len,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    data_args.init_for_training(seed=42)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_predict=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        report_to=args.report_to,
    )
    finetuning_args = FinetuningArguments(stage="mdm", finetuning_type="full")

    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, False, diffusion_args=diff_args)
    dataset = get_dataset(model_args, data_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage=finetuning_args.stage)
    dataset_dict = split_dataset(dataset, data_args, training_args)
    eval_dataset = dataset_dict["eval_dataset"]

    trainer = CustomDiffusionTrainer(
        diff_args=diff_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        eval_dataset=eval_dataset,
    )
    trainer.is_in_train = False
    dataloader = trainer.get_eval_dataloader(eval_dataset)

    prediction_path = output_dir / "generated_predictions.jsonl"
    loss_sum = 0.0
    seen = 0
    correct = 0
    start_time = time.time()

    with prediction_path.open("w", encoding="utf-8") as writer:
        for step, inputs in enumerate(dataloader, start=1):
            inputs = trainer._prepare_inputs(inputs)
            with torch.no_grad():
                loss, pred, label = trainer.prediction_step(
                    trainer.model,
                    inputs,
                    prediction_loss_only=False,
                )
            batch_size = int(label.shape[0])
            batch_correct_count, decoded_preds, decoded_labels = batch_correct(pred, label, tokenizer)
            for decoded_pred, decoded_label in zip(decoded_preds, decoded_labels):
                writer.write(json.dumps({"label": decoded_label, "predict": decoded_pred}, ensure_ascii=False) + "\n")

            seen += batch_size
            correct += batch_correct_count
            if loss is not None:
                loss_sum += float(loss.detach().cpu().item()) * batch_size

            if step == 1 or step % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"RUN_PROGRESS step={step} seen={seen} correct={correct} "
                    f"acc={correct / seen:.8f} elapsed={elapsed:.1f}s "
                    f"samples_per_second={seen / elapsed:.3f}",
                    flush=True,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    runtime = time.time() - start_time
    metrics = {
        "predict_loss": loss_sum / seen if seen else float("nan"),
        "predict_acc": correct / seen if seen else 0.0,
        "predict_runtime": runtime,
        "predict_samples_per_second": seen / runtime if runtime else 0.0,
        "predict_steps_per_second": len(dataloader) / runtime if runtime else 0.0,
        "predict_samples": seen,
        "predict_correct": correct,
    }
    for name in ("predict_results.json", "all_results.json"):
        (output_dir / name).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

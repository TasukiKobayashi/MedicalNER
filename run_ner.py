import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
from model import BertCRF, BertLstmCRF

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    crf: bool = field(
        default=False, metadata={"help": "Add CRF layer to model"}
    )
    lstmcrf: bool = field(
        default=False, metadata={"help": "Add BiLSTM-CRF layer to model"}
    )
    pretraining: Optional[str] = field(
        default=False, metadata={"help": "Pretrained task"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    n_train: Optional[int] = field(
        default=-1, metadata={"help": "# training examples. -1 means use all."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # Support: 1) JSON config file, 2) CLI args, 3) no-arg fallback to sensible repo paths
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse arguments from a json file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # If no CLI args provided, provide a friendly default that points to repo data/output
        if len(sys.argv) == 1:
            repo_dir = Path(__file__).resolve().parent
            default_cli = [
                "--data_dir",
                str(repo_dir / "data"),
                "--model_name_or_path",
                "vinai/phobert-base",
                "--labels",
                str(repo_dir / "data" / "labels.txt"),
                "--output_dir",
                str(repo_dir / "output"),
                "--max_seq_length",
                "128",
                "--num_train_epochs",
                "10",
                "--per_device_train_batch_size",
                "32",
                "--per_device_eval_batch_size",
                "32",
                "--seed",
                "11",
                "--logging_strategy",
                "epoch",
                "--logging_steps",
                "1",
                "--load_best_model_at_end",
                "True",
                "--save_total_limit",
                "2",
                "--save_strategy",
                "no",
                "--do_predict",
                "True",
                "--overwrite_output_dir",
                "True",
            ]
            model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=default_cli)
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast,
    # )
    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    if model_args.crf:
        model = BertCRF.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif model_args.lstmcrf:
        model = BertLstmCRF.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # for param in model.bert.parameters():
    #     param.requires_grad = False
    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False

    # Get datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            n_obs=data_args.n_train,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        if len(predictions.shape) == 3:
            preds = np.argmax(predictions, axis=2)
        else:
            preds = predictions

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        
        return {
            "classification_report": classification_report(out_label_list, preds_list, digits=3),
        }

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # trainer.save_model()

        if model_args.pretraining:
            model.roberta.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model()
            
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w", encoding="utf-8") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, gold_list = align_predictions(predictions, label_ids)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w", encoding="utf-8") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
                with open(os.path.join(data_args.data_dir, "test.txt"), "r", encoding="utf-8", errors="replace") as f:
                    token_classification_task.write_predictions_to_file(writer, f, preds_list)

            # Compute basic per-label metrics and save visualizations
            try:
                # If gold labels are present (non-empty sequences), compute metrics; otherwise skip
                if any(len(s) > 0 for s in gold_list):
                    # Compute per-label precision/recall/f1/support manually from flattened lists
                    y_true = [lab for seq in gold_list for lab in seq]
                    y_pred = [lab for seq in preds_list for lab in seq]
                    labels_arr = labels
                    precisions = []
                    recalls = []
                    f1s = []
                    supports = []
                    for lab in labels_arr:
                        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
                        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
                        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1v = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        support = tp + fn
                        precisions.append(prec)
                        recalls.append(rec)
                        f1s.append(f1v)
                        supports.append(support)
                    df = pd.DataFrame({'label': labels_arr, 'precision': precisions, 'recall': recalls, 'f1': f1s, 'support': supports})
                    df = df.sort_values('f1', ascending=False).reset_index(drop=True)

                    # Save CSV
                    metrics_csv = os.path.join(training_args.output_dir, 'per_label_metrics.csv')
                    df.to_csv(metrics_csv, index=False, encoding='utf-8')

                    # Save JSON summary (including trainer metrics and per-label averages)
                    summary = {
                        'trainer_metrics': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in metrics.items()},
                        'per_label': df.to_dict(orient='records')
                    }
                    metrics_json = os.path.join(training_args.output_dir, 'metrics_summary.json')
                    with open(metrics_json, 'w', encoding='utf-8') as jf:
                        json.dump(summary, jf, ensure_ascii=False, indent=2)

                    # Plot and save figure
                    plt.figure(figsize=(10, max(4, len(df) * 0.25)))
                    sns.set(style='whitegrid')
                    df_melt = df.melt(id_vars=['label'], value_vars=['precision', 'recall', 'f1'], var_name='metric', value_name='score')
                    ax = sns.barplot(data=df_melt, x='score', y='label', hue='metric')
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Score')
                    ax.set_title('Per-label Precision / Recall / F1')
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    metrics_png = os.path.join(training_args.output_dir, 'per_label_metrics.png')
                    plt.savefig(metrics_png, dpi=150)
                    plt.close()
                else:
                    # No gold labels available in test set — write a summary noting this
                    summary = {
                        'trainer_metrics': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in metrics.items()},
                        'note': 'No gold labels found in test set; per-label metrics not computed.'
                    }
                    metrics_json = os.path.join(training_args.output_dir, 'metrics_summary.json')
                    with open(metrics_json, 'w', encoding='utf-8') as jf:
                        json.dump(summary, jf, ensure_ascii=False, indent=2)
                # Confusion matrix (flattened labels)
                try:
                    y_true = [lab for seq in gold_list for lab in seq]
                    y_pred = [lab for seq in preds_list for lab in seq]
                    if len(y_true) != len(y_pred):
                        m = min(len(y_true), len(y_pred))
                        y_true = y_true[:m]
                        y_pred = y_pred[:m]

                    cm = confusion_matrix(y_true, y_pred, labels=list(labels_arr))
                    cm_df = pd.DataFrame(cm, index=list(labels_arr), columns=list(labels_arr))
                    cm_csv = os.path.join(training_args.output_dir, 'confusion_matrix.csv')
                    cm_df.to_csv(cm_csv, encoding='utf-8')

                    plt.figure(figsize=(12, max(6, len(labels_arr)*0.25)))
                    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix (counts)')
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.tight_layout()
                    cm_png = os.path.join(training_args.output_dir, 'confusion_matrix.png')
                    plt.savefig(cm_png, dpi=150)
                    plt.close()
                except Exception:
                    logger.exception('Failed to compute/save confusion matrix')
            except Exception:
                logger.exception('Failed to compute/save metrics/visualization')

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

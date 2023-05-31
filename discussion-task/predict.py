import pandas as pd
from pathlib import Path
import typer
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from datasets import load_dataset, Dataset

IN_COLAB = Path(__file__).parent.name == 'content'  # detect colab run
SCRIPT_FOLDER = Path(__file__).parent
ROOT_FOLDER = SCRIPT_FOLDER if IN_COLAB else SCRIPT_FOLDER.parent
app = typer.Typer(add_completion=False)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE and (not IS_BF16_AVAILABLE)
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)
print('IS_FP16_AVAILABLE', IS_FP16_AVAILABLE)
print('IS_BF16_AVAILABLE', IS_BF16_AVAILABLE)

ds_full_path = f"{SCRIPT_FOLDER}/data/dataset_full.jsonl"
ds_full_d_path = f"{SCRIPT_FOLDER}/data/dataset_full_delta.jsonl"
ds_l0_path = f"{SCRIPT_FOLDER}/data/dataset_lvl0.jsonl"
ds_l0_d_path = f"{SCRIPT_FOLDER}/data/dataset_lvl0_delta.jsonl"

files = [ds_l0_d_path]


def load_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Load dataset from dataframe
    :param df: dataframe with data from file
    :param tokenizer: tokenizer to use
    :return: HF dataset
    """

    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        """Tokenize single example/examples in HF style, DataCollator expected to be applied after"""
        tokenized_examples = tokenizer('' if examples['title'] is None else examples['title'], examples['body'],
                                       truncation=True,
                                       padding='do_not_pad')

        return tokenized_examples

    tokenized_ds = dataset.map(tokenize_function, batched=True)

    return tokenized_ds


def get_trainer_args():
    return TrainingArguments(
        output_dir='tmp',
        report_to='none',
        auto_find_batch_size=False,
        per_device_eval_batch_size=32,
        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_FP16_AVAILABLE,
        fp16_full_eval=IS_FP16_AVAILABLE,
        bf16=IS_BF16_AVAILABLE,
        bf16_full_eval=IS_BF16_AVAILABLE,
    )


@app.command()
def main(
        model_name: str = typer.Option('sara-nabhani/ltp-roberta-large-default',
                                       help='Pretrained model to finetune: HUB or Path'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True,
                                            help='Folder to save results'),
):
    clear_base_model = model_name.replace('/', '-')
    dir_name_to_save = f'ltp-{clear_base_model}-predictions/'
    output_dir = str(results_folder / dir_name_to_save)
    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20,
                                                               problem_type="multi_label_classification")

    # create trainer
    training_args = get_trainer_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    for file in files:
        df = pd.read_json(file, lines=True)
        df['title'] = df.title.apply(lambda x: '' if x is None else x)
        tokenized_dataset = load_dataset(df, tokenizer)

        print('\n', '-' * 32, 'Predicting', '-' * 32, '\n')
        test_prediction = trainer.predict(tokenized_dataset)
        predictions = test_prediction.predictions

        pred_df = pd.DataFrame(predictions, columns=list(model.config.id2label.values()))
        df = pd.concat([df, pred_df], axis=1)
        file_name = file.split('/')[-1]
        df.to_csv(f'{output_dir}{file_name[:-6]}.csv', index=False)


if __name__ == '__main__':
    app()



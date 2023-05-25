import os
import ast
import json
import random
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
import typing as tp

import pandas as pd
import typer
from datasets import load_dataset, Dataset, DatasetDict, Sequence, ClassLabel, Value
import torch
from nlpaug import Augmenter
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from scipy.special import expit
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, EarlyStoppingCallback, PreTrainedTokenizer
import evaluate
import numpy as np
from torchinfo import summary
import nltk
from nltk.corpus import stopwords
import gensim.downloader as api
from transformers.integrations import NeptuneCallback
import neptune.new as neptune


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

IN_COLAB = Path(__file__).parent.name == 'content'  # detect colab run
SCRIPT_FOLDER = Path(__file__).parent
ROOT_FOLDER = SCRIPT_FOLDER if IN_COLAB else SCRIPT_FOLDER.parent
MODELS_DIR = f"{ROOT_FOLDER}/models/"
with open(SCRIPT_FOLDER / 'params.json') as f:
    EDOS_EVAL_PARAMS = json.load(f)


# prefer bf16, https://www.reddit.com/r/MachineLearning/comments/vndtn8/d_mixed_precision_training_difference_between/
IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE and (not IS_BF16_AVAILABLE)
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)
print('IS_FP16_AVAILABLE', IS_FP16_AVAILABLE)
print('IS_BF16_AVAILABLE', IS_BF16_AVAILABLE)


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stops = list(stopwords.words('english'))

app = typer.Typer(add_completion=False)


class ValueEval2023Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, aug_type: tp.Optional[str] = None, dataset_folder: tp.Optional[Path] = SCRIPT_FOLDER / 'data', include_stance: bool = False, samples_per_class: tp.Optional[int] = None):
        """
        Load dataset from files
        :param split: either "train", "validation", "test"
        :param tokenizer: HF tokenizer to apply online while sampling with `self[id]`
        :param aug_type: None, "uca" (original article) TODO: add augmentation
        :param dataset_folder: folder with dataset files (arguments-*.tsv, labels-*.tsv)
        :param include_stance: whatever include stance in the tokenized dataset
        :param samples_per_class: if not None, select k samples per class, class can intersect
        """
        # Load dataset
        self.split = split
        self.samples_per_class = samples_per_class
        self.dataset, self.class_names, self.ids = self._load_hf_dataset(dataset_folder, split, samples_per_class=samples_per_class)
        self.tokenizer = tokenizer
        self.aug_type = aug_type
        self.include_stance = include_stance

    @staticmethod
    def _select_samples_per_class(df_labels: pd.DataFrame, samples_per_class: int) -> Dataset:
        """Select k samples per class"""
        # for each columns select k samples
        select_indexes = set()
        for cl in df_labels.drop('id', axis=1).columns.tolist():
            _class_indexes = np.where(np.array(df_labels[cl]) == 1)[0]
            _class_indexes = np.random.choice(_class_indexes, size=samples_per_class, replace=False)
            select_indexes.update(_class_indexes)

        # select samples
        df_labels = df_labels.iloc[list(select_indexes)]

        return df_labels

    @staticmethod
    def _load_hf_dataset(dataset_folder: Path, split: str, samples_per_class: tp.Optional[int] = None) -> tp.Tuple[Dataset, tp.List[str], tp.List[str]]:
        """Load dataset from files
        :param dataset_folder: folder with dataset files (arguments-*.tsv, labels-*.tsv)
        :param split: eather "train", "validation", "test"
        :param samples_per_class: if not None, select k samples per class, class can intersect
        :return: HF dataset with columns ['id', 'premise', 'conclusion', 'stance', 'labels'] and class names list
        """
        _df_labels = pd.read_csv(dataset_folder / f'labels-{split}.tsv', sep='\t')
        _df_labels = _df_labels.rename(columns={'Argument ID': 'id'})
        if samples_per_class is not None:
            _df_labels = ValueEval2023Dataset._select_samples_per_class(_df_labels, samples_per_class)
        class_names = _df_labels.drop('id', axis=1).columns.tolist()
        _df_labels['labels'] = _df_labels.drop('id', axis=1).values.tolist()
        _df_labels = _df_labels[['id', 'labels']]
        # make labels list of floats as suggested here
        # https://discuss.huggingface.co/t/fine-tune-for-multiclass-or-multilabel-multiclass/4035/23?page=2
        _df_labels['labels'] = _df_labels['labels'].apply(lambda x: [float(i) for i in x])

        _df_texts = pd.read_csv(dataset_folder / f'arguments-{split}.tsv', sep='\t')
        _df_texts = _df_texts.rename(columns={'Argument ID': 'id', 'Conclusion': 'conclusion', 'Premise': 'premise', 'Stance': 'stance'})

        _df = _df_labels.merge(_df_texts, on='id')

        dataset = Dataset.from_pandas(_df)
        # dataset.cast_column('labels', feature=Sequence(Value(dtype='int64'), length=20))

        return dataset, class_names, _df['id'].tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def _tokenize_function(self, examples: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        """Tokenize single example/examples in HF style, DataCollator expected to be applied after"""
        if self.include_stance:
            #  TODO: add stance
            raise NotImplementedError('yet. conclusion [sep] stance [sep] premise')
        else:
            tokenized_examples = self.tokenizer(examples['conclusion'], examples['premise'], truncation=True, padding='do_not_pad')

        return {
            'labels': examples['labels'],
            **tokenized_examples
        }

    def _check_download_model(self, model_name: str):
        """Check if word embedding models exist and download if not"""
        if os.path.exists(f"{MODELS_DIR}{model_name}.bin"):
            print("Model exists")
        else:
            print(f'Downloading {model_name} model...')
            model = api.load(model_name)
            model.save_word2vec_format(f"{MODELS_DIR}{model_name}.bin", binary=True)

    def _get_augmenter(self, aug_type: str):
        """Select Augment per aug_type"""
        if aug_type == 'uca':
            # Select augmentation type to apply
            aug_class: Augmenter = random.choices(
                [
                    nac.RandomCharAug(action='delete', aug_char_p=0.15, stopwords=stops),
                    nac.RandomCharAug(action='swap', aug_char_p=0.05, stopwords=stops),
                    naw.RandomWordAug(action='delete', aug_p=0.15, stopwords=stops),
                    naw.RandomWordAug(action='swap', aug_min=2, aug_max=2, stopwords=stops)
                ],
                weights=(10, 10, 40, 40),  # probs of selecting each augmenter
                k=1
            )[0]
        # noising char
        elif aug_type == 'char_ins':
            aug_class = nac.RandomCharAug(action="insert", stopwords=stops)
        elif aug_type == 'char_del':
            aug_class = nac.RandomCharAug(action="delete", stopwords=stops)
        elif aug_type == 'char_sub':
            aug_class = nac.RandomCharAug(action="substitute", stopwords=stops)
        elif aug_type == 'char_swap':
            aug_class = nac.RandomCharAug(action="swap", stopwords=stops)
        elif aug_type == 'char_key':
            aug_class = nac.KeyboardAug(stopwords=stops)
        # noising word
        elif aug_type == 'word_crop':
            aug_class = naw.RandomWordAug(action="crop", stopwords=stops)
        elif aug_type == 'word_del':
            aug_class = naw.RandomWordAug(action="delete", stopwords=stops)
        elif aug_type == 'word_sub':
            # TODO choose what words to use for substitution
            aug_class = naw.RandomWordAug(action="substitute", target_words=['is', 'are', 'the', 'you', 'was', 'were', 'am'], stopwords=stops)
        elif aug_type == 'word_swap':
            aug_class = naw.RandomWordAug(action="swap", stopwords=stops)
        elif aug_type == 'word_split':
            aug_class = naw.SplitAug(stopwords=stops)
        elif aug_type == 'word_spell':
            aug_class = naw.SpellingAug(stopwords=stops)
        elif aug_type == 'word_syn':
            aug_class = naw.SynonymAug(stopwords=stops)
        elif aug_type == 'word_emb_w2v':
            self._check_download_model('word2vec-google-news-300')
            aug_class = naw.WordEmbsAug(model_type='word2vec', model_path=f"{MODELS_DIR}word2vec-google-news-300.bin",
                                        action="substitute", stopwords=stops)
        elif aug_type == 'word_emb_glv':
            self._check_download_model('glove-wiki-gigaword-100')
            aug_class = naw.WordEmbsAug(model_type='glove', model_path=MODELS_DIR+'glove-wiki-gigaword-100.bin',
                                        action="substitute", stopwords=stops)
        elif aug_type == 'word_emb_ft':
            self._check_download_model('fasttext-wiki-news-subwords-300')
            aug_class = naw.WordEmbsAug(model_type='fastest', model_path=MODELS_DIR+'fasttext-wiki-news-subwords-300.bin',
                                        action="substitute", stopwords=stops)
        elif aug_type == 'word_con_emb_roberta':
            aug_class = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", stopwords=stops, device='cuda')
        elif aug_type == 'word_con_emb_bert':
            aug_class = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", stopwords=stops, device='cuda')
        elif aug_type == 'word_con_emb_distilbert':
            aug_class = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute", stopwords=stops, device='cuda')
        elif aug_type == 'word_con_emb_squeezebert':
            aug_class = naw.ContextualWordEmbsAug(model_path='squeezebert/squeezebert-uncased', action="substitute", stopwords=stops, device='cuda')
        elif aug_type == 'word_con_emb_bart':
            aug_class = naw.ContextualWordEmbsAug(model_path='facebook/bart-base', model_type='bart', action="substitute", stopwords=stops, device='cuda')
        elif aug_type == 'word_backtranslate':
            aug_class = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-de', to_model_name='Helsinki-NLP/opus-mt-de-en', device='cuda')
        elif aug_type == 'sen_con_emb_gpt2':
            aug_class = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2', device='cuda')
        elif aug_type == 'sen_con_emb_distilgpt2':
            aug_class = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2', device='cuda')
        elif aug_type == 'sen_random':
            aug_class = nas.RandomSentAug('random')
        elif aug_type == 'sen_summ':
            aug_class = nas.AbstSummAug('t5-small', device='cuda')
        elif self.aug_type is None:
            return None
        elif len(aug_type.split(':')) > 1:
            aug_t = aug_type.split(':')[0] # random or all
            augs = aug_type.split(':')[1] # list of augmenters
            augmenters = [self._get_augmenter(aug.strip()) for aug in augs.split(',')]
            if aug_t == 'random':
                aug_class = naf.Sometimes(augmenters)
            elif aug_t == 'all':
                aug_class = naf.Sequential(augmenters)
            else:
                raise NotImplementedError(f'Augment {self.aug_type} not implemented')
        else:
            raise NotImplementedError(f'Augment {self.aug_type} not implemented')

        return aug_class

    def _augment_single_sent(self, text: str) -> tp.Tuple[str, str]:
        """Augment single example"""

        aug_class = self._get_augmenter(self.aug_type)
        if aug_class is None:
            return text
        else:
            aug_text = aug_class.augment(text)[0]
            return aug_text

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]

        if self.aug_type:
            sample['conclusion'] = self._augment_single_sent(sample['conclusion'])
            sample['premise'] = self._augment_single_sent(sample['premise'])

        sample = self._tokenize_function(sample)

        return sample


def _get_metrics_function():
    metric_f1 = evaluate.load('f1', 'multilabel')

    def _compute_metrics(eval_preds):
        preds, labels = eval_preds
        probs = expit(preds)
        predictions = np.where(probs > 0.5, 1, 0)
        labels = labels.astype(int)
        return {
            **metric_f1.compute(predictions=predictions, references=labels, average='macro'),
            'f1_all': metric_f1.compute(predictions=predictions, references=labels, average=None)['f1'].tolist()
        }

    return _compute_metrics


def _get_trainer_args(params, hub_model_name, output_dir, push_to_hub=False):
    return TrainingArguments(
        output_dir=output_dir,
        report_to='none',

        learning_rate=params['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=params.get('weight_decay', 0.01),
        optim=params.get('optim', 'adamw_torch'),

        auto_find_batch_size=False,  # divide by 2 in case of OOM
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['max_epochs'],
        warmup_ratio=params.get('warmup_ratio', 0.05),

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_FP16_AVAILABLE,
        fp16_full_eval=IS_FP16_AVAILABLE,
        bf16=IS_BF16_AVAILABLE,
        bf16_full_eval=IS_BF16_AVAILABLE,

        logging_strategy='steps',
        logging_steps=params['eval_steps'],
        evaluation_strategy='steps',
        eval_steps=params['eval_steps'],
        save_strategy='steps',
        save_steps=params['eval_steps'],

        metric_for_best_model='f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,

        torch_compile=False,  # not working as Tesla T4 for now

        hub_model_id=hub_model_name,
        push_to_hub=push_to_hub,
        hub_strategy='checkpoint',
    )


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='Pretrained model to finetune: HUB or Path'),
        config_name: str = typer.Option('default', help='Config name to use: see params.json'),
        aug_type: str = typer.Option(None, help='Argumentation type to use'),
        postfix: str = typer.Option('', help='Model name postfix'),
        push_to_hub: bool = typer.Option(False, help='Push model to HuggingFace Hub'),
        save_model: bool = typer.Option(False, help='Save model locally'),
        skip_neptune: bool = typer.Option(False, help='Skip neptune metrics tracking (for debug)'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    clear_base_model = base_model.replace('/', '-')
    model_name_to_save = f'ltp-{clear_base_model}-{config_name}'
    if postfix:
        model_name_to_save += f'{model_name_to_save}-{postfix}'
    output_dir = str(results_folder / model_name_to_save)
    model_save_folder = save_folder / model_name_to_save
    hub_model_name = f'k4black/{model_name_to_save}'

    # load config
    params = EDOS_EVAL_PARAMS[config_name.split('-')[0]]  # read base config
    params.update(EDOS_EVAL_PARAMS[config_name])  # update with specific config
    #aug_type = params.get('aug_type', None)
    samples_per_class = params.get('samples_per_class', None)
    num_folds = params.get('num_folds', 1)

    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # create neptune run
    if not skip_neptune:
        neptune_run = neptune.init_run(tags=['task:finetuning', f'folds:{num_folds}', f'aug:{aug_type}', f'k:{samples_per_class}', f'model:{base_model}', f'conf:{config_name}'])
        neptune_object_id = neptune_run['sys/id'].fetch()
        print('neptune_object_id', neptune_object_id)
        neptune_run['finetuning/parameters'] = {
            'base_model': base_model,
            'config_name': config_name,
            'aug_type': aug_type,
            'samples_per_class': samples_per_class,
            'folds': num_folds,
        }
        neptune_run.stop()
    else:
        print('Skip neptune run')

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load metrics
    compute_metrics = _get_metrics_function()

    folds_metrics = defaultdict(list)
    for i in range(num_folds):
        print('\n', '-' * 32, f'Fold {i + 1}/{num_folds}', '-' * 32, '\n')

        # fold neptune run - restored old one
        if not skip_neptune:
            fold_neptune_run = neptune.init_run(with_id=neptune_object_id)
            folds_neptune_callback = NeptuneCallback(run=fold_neptune_run, base_namespace=f'fold-{i+1}')

        # load data, inside each fold to keep selecting random [samples_per_class] samples each fold

        train_dataset = ValueEval2023Dataset('train', tokenizer, aug_type=aug_type, samples_per_class=samples_per_class, include_stance=False)
        val_dataset = ValueEval2023Dataset('validation', tokenizer)
        test_dataset = ValueEval2023Dataset('test', tokenizer)
        print(f'train_dataset: {len(train_dataset)} \t val_dataset: {len(val_dataset)} \t test_dataset: {len(test_dataset)}')

        # load new fold pretrained model
        # config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
        # model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, ignore_mismatched_sizes=True, num_labels=20, problem_type='multi_label_classification')
        summary(model)

        # create trainer
        training_args = _get_trainer_args(params, hub_model_name, output_dir, push_to_hub=push_to_hub)
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=params.get('early_stopping_patience', 5)),
        ]
        if not skip_neptune:
            callbacks.append(folds_neptune_callback)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        print('\n', '-' * 32, f'Fold {i + 1}/{num_folds} Training...', '-' * 32, '\n')

        # train itself
        trainer.train()

        # save model
        if save_model:
            if model_save_folder:
                model_save_folder.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(model_save_folder))

        print('\n', '-' * 32, f'Fold {i + 1}/{num_folds} End', '-' * 32, '\n')

        final_metrics = {}
        for metric_key_prefix, ds in [('eval', val_dataset), ('test', test_dataset)]:
            print(f'Evaluating {metric_key_prefix} split dataset...')

            ds_prediction = trainer.predict(ds, metric_key_prefix=metric_key_prefix)

            fold_metrics = {
                f'{metric_key_prefix}_f1': ds_prediction.metrics[f'{metric_key_prefix}_f1'],
                f'{metric_key_prefix}_f1_all': ds_prediction.metrics[f'{metric_key_prefix}_f1_all'],
                **{f'{metric_key_prefix}_f1_{class_name}': f1_score for class_name, f1_score in zip(ds.class_names, ds_prediction.metrics[f'{metric_key_prefix}_f1_all'])}
            }
            final_metrics.update(fold_metrics)
            print(metric_key_prefix, 'fold_metrics', fold_metrics)

            if not skip_neptune:
                for k, v in fold_metrics.items():
                    if 'f1_all' not in k:
                        folds_neptune_callback.run[f'finetuning/folds_metrics/{k}'].append(v)

                # save predictions file to neptune
                prob_predictions = expit(ds_prediction.predictions)
                df = pd.DataFrame(prob_predictions, columns=ds.class_names, index=ds.ids)
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=True, index_label='id')
                folds_neptune_callback.run[f'predictions/fold-{i+1}-{metric_key_prefix}'].upload(neptune.types.File.from_stream(csv_buffer, extension="csv"))

        if not skip_neptune:
            folds_neptune_callback.run[f'fold-{i+1}/final_metrics'] = final_metrics
            folds_neptune_callback.run.stop()

        time.sleep(5)

        for k, v in final_metrics.items():
            folds_metrics[k].append(v)

    print('\n', '-' * 32, 'End', '-' * 32, '\n')

    average_fold_final_metrics = {
        k: np.mean(v) if 'f1_all' not in k else np.mean(v, axis=0).tolist()
        for k, v in folds_metrics.items()
    }
    print('folds_metrics', folds_metrics)
    print('average_fold_final_metrics', average_fold_final_metrics)
    if not skip_neptune:
        neptune_run = neptune.init_run(with_id=neptune_object_id)
        neptune_run['finetuning/final_metrics'] = average_fold_final_metrics


if __name__ == '__main__':
    app()

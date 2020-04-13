# Unsupervised Opinion Summarization as Copycat-Review Generation

This repository contains the Python (PyTorch) codebase of [corresponding paper](https://arxiv.org/abs/1911.02247
) accepted at ACL 2020, Seattle, USA.


<p align="center">
<img src="img/diagram.png" width="300">
</p>

The model is fully **unsupervised** and is trained on a large corpus of customer reviews, such as Yelp or Amazon. It generates **abstractive** summaries condensing common opinions across a group of reviews.  It relies on Bayesian auto-encoding that fosters learning rich hierarchical semantic representations of reviews and products. Finally, the model uses a copy mechanism to better preserve details of input reviews.

Example summaries produced by the system are shown below.


* *This restaurant is a hidden gem in Toronto. The food is delicious, and the service is impeccable. Highly recommend for anyone who likes French bistro.*

* *This is a great case for the Acer Aspire 14" laptop. It is a little snug for my laptop, but it's a nice case. I would recommend it to anyone who wants to protect their laptop.*

* *This is the best steamer I have ever owned. It is easy to use and easy to clean. I have used it several times and it works great. I would recommend it to anyone looking for a steamer.*


For more examples, please refer to the [artifacts folder](copycat/artifacts/yelp/summs).

## Installation

The easiest way to proceed is to create a separate [conda environment](https://docs.conda.io/en/latest/).

```
conda create -n copycat python=3.6.9
```

```
conda activate copycat
```

Install required modules.

```
pip install -r requirements.txt
```

Add the root directory to the path.

```
export PYTHONPATH=root_path:$PYTHONPATH
```

### Notes

* Minor deviations from the published results are expected as the code was migrated from a bleeding-edge PyTorch version and Python 2.7.

* Post factum, we added a **beam search generator** that has the **n-gram blocking functionality** (based on OpenNMT). The enhancement allows for a repetition reduction.

* The setup was fully tested with **Python 3.6.9**.

* The model work on a single GPU only.

* **mltoolkit** provides the backbone functionality for data processing and modelling. Make sure it's visible to the interpreter.

## Data

Our model is trained on two different collections of customer reviews - [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets.html) and [Yelp](https://www.yelp.nl/dataset/challenge). The evaluation was performed on human-annotated summaries based on both datasets.

### Preprocessing of Unsupervised Data
To train the model, one needs to download the datasets from the official websites. Both are publicly available, free of charge.
The model expects a certain format of input, which can be obtained by preprocessing the downloaded data using the [provided preprocessing scripts](preprocessing/).

### Input Data Format

If training should be performed on a separate dataset, the expected format of input is provided in [artifacts](artifacts/amazon/data/input_example). Each business/product has to be separated to CSV files where each line corresponds to a separate review.


group_id | review_text | rating | category
--- | --- | --- | ---
159985130X | We recommend the Magnifier ...  | 4.0 | health_and_personal_care

The rating column is optional as it is not used by the model.

### Evaluation Summaries

Evaluation can be performed on human-created summaries, both [Amazon](https://github.com/ixlan/CopyCat-abstractive-Amazon-product-summaries) and [Yelp](https://github.com/sosuperic/MeanSum) summaries are publicly available. No preprocessing is needed for evaluation.

## Running

If you preprocessed data yourself, please create your vocabulary and truecaser. Otherwise, you can skip the following two sections.

### Vocabulary Creation

Vocabulary contains to a mapping from words to frequency, where file position corresponds to ids used by the model.

```
python copycat/scripts/create_vocabulary.py --data_path=your_data_path --vocab_fp=data/dataset_name/vocabs/vocab.txt
```

### Truecaser Creation
Truecaser is used to reverse lowercase letters, and needs to be trained (quickly) by scanning the dataset. Note that multiple folders can be assigned to the `data_path` parameter.

```
python copycat/scripts/train_truecaser.py --data_path=your_data_path --tcaser_fp=data/dataset_name/tcaser.model
```

### Workflow

One needs to set parameters of the workflow in `copycat/hparams/run_hp.py`. E.g., by altering data paths or specifying the number of training epochs.

The file `run_copycat.py` contains a workflow of operations that are executed to prepare necessary objects (e.g., beam search) and then run a training and/or evaluation procedure.
After adjusting run parameters, execute the following command.

```
python copycat/scripts/run_workflow.py
```


### Checkpoints

[Amazon](https://drive.google.com/open?id=143BhjMPL5vdNdjk0-duAz4LBB7FBVhXx) and [Yelp](https://drive.google.com/open?id=1wy8lpokZqf3KygQQJTLrPVT7q6Ok3Hgr) checkpoints are available for download. Please them to `copycat/artifacts/` to the corresponding sub-folders.

## LICENSE

MIT


## Citation

(will be updated soon)

```
@article{bravzinskas2019unsupervised,
  title={Unsupervised Multi-Document Opinion Summarization as Copycat-Review Generation},
  author={Bra{\v{z}}inskas, Arthur and Lapata, Mirella and Titov, Ivan},
  journal={arXiv preprint arXiv:1911.02247},
  year={2019}
}
```

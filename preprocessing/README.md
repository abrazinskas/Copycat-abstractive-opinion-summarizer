# Instructions
The preprocessing script is based on [Luigi workflows](https://github.com/spotify/luigi), which parallels preprocessing steps over files to achieve a speed-up. The script is adjusted to work for [Amazon](http://jmcauley.ucsd.edu/data/amazon/links.html) and [Yelp](https://www.yelp.nl/dataset/challenge) data.
<img src="../img/luigi.png" width="200">



Overall, the preprocessing workflow executes the following logic:

1. **Preparation** - groups each product/business reviews to separate CSV files that have proper columns.
2. **Tokenization** - tokenizes sequences with the Moses (reversible) tokenizer.
3. **Subsampling** - filters too short reviews, very unpopular and too popular products/businesses.
4. **Partitioning** - assigns the groups of reviews to training validation partitions.

The output of each step is used as input to a subsequent step, and thus no re-computation is needed of upstream steps if the downstream step needs to be recomputed (e.g., if subsampling parameters change).

## Raw Data

### Amazon

1. Create a folder, e.g, *data/amazon/raw/*.

2. [Download](http://jmcauley.ucsd.edu/data/amazon/links.html) reviews of products of some categories; do not unzip the files. E.g., you might end up with the following files: *Clothing_Shoes_and_Jewelry.json.gz*,
*Electronics.json.gz*, *Health_and_Personal_Care.json.gz*, *Home_and_Kitchen.json.gz*.

3. Place the files to the created folder.


### Yelp

1. Create a folder for data, e.g., *data/yelp/raw/*.
2. Download [Yelp](https://www.yelp.nl/dataset/challenge) and unzip it.
3. It comes with a bundle of metafiles, while we need only the customer reviews. Put **only the file with reviews** to the created folder, name it *reviews.json*.

## Running the Workflow

Before starting, please make sure that the root directory is visible by the interpreter. For example, by adding it to the path environmental variable:

```
export PYTHONPATH=root_path:$PYTHONPATH
```

From the root directory execute the following command (undo the formatting).

```
python -m luigi
--local-scheduler
--module preprocessing.steps Partition
--inp-dir-path=../data/amazon/raw
--Subsample-min-revs=10
--Subsample-min-rev-len=20
--Subsample-max-rev-len=70
--Subsample-percentile=90
--train-part=0.95
--val-part=0.05
--GlobalConfig-out-dir-path=../data/
--GlobalConfig-dataset=amazon
--log-level=INFO
--workers=4
```


The most important parameters are:
* **input-dir-path** - directory with data.
* **Subsample-min-revs** - minimum number of reviews each group should have.
* **Subsample-min-rev-len** - maximum number of tokens the review should have.
* **Subsample-percentile** - threshold percentile of reviews; groups with a larger number of reviews are discarded.
* **train-part** - the proportion (in [0.,1.]) of reviews that should be part of the training set.
* **val-part** - the proportion (in [0., 1.]) of reviews that should be part of the validation set.
* **workers** - total number of workers to assign to the workflow. Each worker handles a separate file. So
* **GlobalConfig-dataset** - yelp or amazon.

The script should create the necessary folders and perform the full procedure to preprocess data. For the model's training, you will need the output located in the **4.part** folder, which corresponds to the partitioned data.


### Notes

* Tokenization is performed using a reversible tokenizer Moses, and it is relatively time-consuming to tokenize a large amount of data. Preprocessing can take up to a day on the full Amazon dataset.

* Tokenization is not multi-processing at the moment of writing, which adds to the preprocessing time.
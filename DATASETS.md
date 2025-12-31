# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```bash
$DATA/
|–– semi-aves/
|–– stanford_cars/
|–– fgvc-aircraft/
|–– eurosat/
|–– dtd/

$RETRIEVED/
|–– semi-aves/
|–– stanford_cars/
|–– fgvc-aircraft/
|–– eurosat/
|–– dtd/
```

Update the `config.yml` with the path to the datasets and retrieved data.

```yaml
dataset_path: /scratch/dataset/
retrieved_path: /scratch/retrieved/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:

- [Semi-Aves](#semi-aves)
- [StanfordCars](#stanfordcars)
- [FGVC-Aircraft](#fgvc-aircraft)
- [EuroSAT](#eurosat)
- [DTD](#dtd)

The instructions to prepare each dataset are detailed below. 


### Semi-Aves

- Create a folder named `semi-aves/` under `$DATA`.
- Download data from the [official repository](https://github.com/cvl-umass/semi-inat-2020).


- The annotations are extracted from the [official annotation json files](https://github.com/cvl-umass/semi-inat-2020). We have reformatted and provided to you as `ltrain.txt`, `ltrain+val.txt`,`val.txt` and `test.txt` in the `SWIFT/data/semi-aves/` folder.
  
The directory structure should look like:

```bash
semi-aves/
|–– trainval_images
|–– u_train_in
|–– test
```

### StanfordCars

- Create a folder named `stanford_cars/` under `$DATA`.
- In case the following link breaks, download dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).
- Download `car_devkit.tgz`

```bash
wget https://github.com/pytorch/vision/files/11644847/car_devkit.tgz
tar -xzvf car_devkit.tgz
```

- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like

```bash
stanford_cars/
|–– cars_test/
|–– cars_annos.mat
|–– cars_train/
|–– split_zhou_StanfordCars.json
```

### FGVC-Aircraft

- Create a folder named `fgvc-aircraft/` under `$DATA`.
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
```

- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `fgvc-aircraft-2013b/data/` to `$DATA/fgvc-aircraft`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWIFT/data/fgvc-aircraft/` folder.

The directory structure should look like:

```bash
fgvc_aircraft/
|–– fgvc-aircraft-2013b/
    |–– data/
```


### EuroSAT

- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from [link](http://madm.dfki.de/files/sentinel/EuroSAT.zip) and extract it to `$DATA/eurosat/`.

```bash
wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
```

- Rename the extracted folder `2750` to `EuroSAT_RGB`.
- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWIFT/data/eurosat/` folder.

The directory structure should look like:

```bash
eurosat/
|–– EuroSAT_RGB/
```

### DTD

- Create a folder named `dtd/` under `$DATA`.
- Download the dataset from [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) and extract it to `$DATA`. This should lead to `$DATA/dtd/`.

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
```

- We have reformatted labels and provided to you as `train.txt`, `val.txt` and `test.txt` in the `SWIFT/data/dtd/` folder.

The directory structure should look like:

```bash
dtd/
|–– dtd/
    |–– images/
    |–– imdb/
    |–– labels/
```
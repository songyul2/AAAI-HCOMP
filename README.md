# Description
This repo is for the AAAI HCOMP 2024 paper [Combining Human and AI Strengths in Object Counting under Information Asymmetry](https://doi.org/10.1609/hcomp.v12i1.31603). The specific task is to combine the predictions of humans and AI to improve the accuracy of object counting.
The project consists of several components, including generating jar images, running the segmentation model on images, performing transformations and predictions for AI, making predictions for humans, and combining the predictions.

# Environment
The code was tested on Linux.
You can use docker to set up the environment . The second command maps $(pwd)/app on your computer to the /app folder inside the container. Any files the script saves will appear on your file system.
```
docker build -t stan-app .
docker run --rm -v "$(pwd)/app:/app" stan-app
```
Alternatively, to do it yourself, first install cmdstanpy following https://mc-stan.org/cmdstanpy/installation.html. Conda installation only worked in an empty conda environment. Another possible issue is discussed at https://discourse.mc-stan.org/t/error-installing-cmdstan/19216/3. Then, run the following command to install the rest of the packages:
`pip install -r requirements.txt`

# Major components
`human-data/jarstudy4.csv` is the behavorial data. Annotators looked at and annotated various angles for the images we provided them. `utils.read_jarstudy4` parses the relevant columns in it.
`./blender` contains the code to generate jar images which works in Linux and Windows. The image files are too large to be included.


`segment-code/segment.py` is the main script that runs the segmentation model on the images, and it saves the output to the directory `density_{subset}`. subset is either `train` or `test`. The directory contains the following files: images with masks and `img_data.csv`.
The `img_data.csv` file contains the following:
> img,id,shapeIndex,angle,estimate,trueNum
Cylinder1_QQJOJR0XS_view1,QQJOJR0XS,Cylinder1,90,186,74

`train_ai.py` is the script that performs transformations and predictions for AI. The script reads the data from the directory `density_{subset}/img_data.csv`, performs transformations, and saves the results to the directory `data/machine-data`.

It utilizes the `train_ai.transform_and_save` function to transform the segmentation model output using regression. The transformed estimates replace the original data and are saved under the directory `data/sam/density_{subset}/img_data.csv`.
The file `preprocessed-train.tsv` is then read by `train_ai.cmdstan` to learn the parameters of the generating model.
<!-- The `utils.preprocess` function is responsible for preprocessing the training and test data. It takes the input data from the directory `train_test_dir / f"density_{subset}/img_data{fold}.csv"`, performs the necessary preprocessing steps, and saves the results to the directory `results_dir / f"preprocessed-{subset}.tsv"`. -->

The additional preprocessing steps include taking the log of the counts or standardizing the counts. For the AI, the log transformation occurs before regression, while for humans, it happens during the `utils.preprocess` step. Standardization is performed during `utils.preprocess` for both the AI and humans. The mean and variance of the ground truth are computed using the AI training data and then applied to the test data and human data.

`join.py` is the script that combines the predictions. Its commandline argument is the fold number 0-4. For fold 0, it reads the data from the directory `data/machine-data` and `data/human-data0`, combines the estimates, and saves the results to the directory `data/human-data0`. Computations for different folds can be parallelized.

`draw.py` is the script that generates the plots.

# Running the code
`. cross-validate.sh`

This assumes that `parallel` is available on the system. Otherwise, you can run the commands sequentially. 
```
python train_ai.py 
python  join.py 0
python  join.py 1
python  join.py 2
python  join.py 3
python  join.py 4
python draw.py 
```
If there was an error saying that a file does not exist, creating a directory with the appropriate name should fix the issue. 

# Output
The internal output is saved under the `data` directory. `draw.py` generates the plots and saves them in the `plots` directory. It also prints the relative improvement in a format for latex tables.

# Project Structure
There are 5 folds, and each fold has a corresponding directory in the `data` directory. For example, 
```data/
├── human-data0/
│   ├── calibrated.tsv
│   ├── parameters-Cylinder1.tsv
│   ├── parameters-Cylinder2.tsv
│   ├── parameters-Disk1.tsv
│   ├── parameters-Disk2.tsv
│   ├── parameters-Sphere1.tsv
│   ├── parameters-Sphere2.tsv
│   ├── preprocessed-test.tsv
│   ├── preprocessed-train.tsv
├── human-data1/
│   ├── calibrated.tsv
│   ├── parameters-Cylinder1.tsv
│   ├── parameters-Cylinder2.tsv
│   ├── parameters-Disk1.tsv
│   ├── parameters-Disk2.tsv
│   ├── parameters-Sphere1.tsv
│   ├── parameters-Sphere2.tsv
│   ├── preprocessed-test.tsv
│   ├── preprocessed-train.tsv
```


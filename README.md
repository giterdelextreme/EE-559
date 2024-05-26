# EE-559 Project

## Model
[MEGA: Moving Average Equipped Gated Attention](https://huggingface.co/docs/transformers/main/model_doc/mega)

## Datasets
- [Large-Scale Hate Speech Detection with Cross-Domain Transfer](https://github.com/avaapm/hatespeech)

- [Hatemoji: A Test Suite and Adversarially-Generated Dataset for Benchmarking and
Detecting Emoji-based Hate](https://github.com/HannahKirk/Hatemoji)

- [Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master)

## Description of each files 
data/tweets.csv : The data set of "Automated Hate Speech Detection and the Problem of Offensive Language"

model/* THe model with and without our cusom loss function when trained on the dataset of "Automated Hate Speech Detection and the Problem of Offensive Language"

outpus/* The SCITAS console outputs of respectively the schduler search, the learning rate search and the lambda of the cusom loss function search

plots/*.obj The savec files from the schduler search, the learning rate search and the lambda of the cusom loss function search

plots/nice_plots.ipynb A notebook used to make some of the plot for the report/poster

MEGA_Hatemoji.ipynb The basic training notebook for the Hatemoji dataset

MEGA_Hatemoji_custom_loss.ipynb The training notebook for the Hatemoji dataset with our custom loss function

MEGA_Hatemoji_threshold.ipynb A tentative of an other alternative to reduce false negative (Deprecated)

MEGA_Tweet.ipynb A notebook used for informaly testing different ideas

MEGA_confusion.py The script used to train and compute the confusion matrix with the cross entropy loss

MEGA_confusion_custom_loss.py The script used to train and compute the confusion matrix with the custom loss

MEGA_rate_finder.py The script used to train and find the optimal learning rate

MEGA_scheduler_finder.py The script used to train and find the optimal scheduler 

main.sh Bash script to run our long trainings on the Scitas cluster

mega_hatemoji.obj contain the metric and model trained on the Hatemoji dataset

mega_hatemoji_seeds.obj contain the metric and model trained on the Hatemoji dataset with our custom loss

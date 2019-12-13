# Music Genre Classification using SVM

A Machine Learning project to predict the genre of the specific song snippet.

## Problem

Given a segment of a song, the task is to classify the genre of the song into one of these classes:
- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Dataset 

GTZAN Genre Collection: The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

Link: [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)

## Preprocessing

### Dataset Splitting

- The original GTZAN dataset contains 1k 30-seconds .wav files.
- Each class has 100 30-seconds snippets from songs
- Used 90% of the data for training (900 sample).
- Used 10% for testing (100 sample)
- Each class has the same number of samples in training and testing (90 and 10 respectively)

### Feature Extraction 

Extracted Features of audio files using [Mel Frequence Cepstrum Coefficients](https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9)

Used Librosa library for feature extraction Link: [Librosa_features](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram)

## Training Method and Results

Files are having following functionalities 

-	DownloadData.py – This file will download the data and save it in genre folder in the current directory (data will be downloaded in .zip format)
-	ExtractData.py – This file will extract the downloaded data and save it in /genre dir 
-	PreprocessData.py - This file will preprocess the data, it will separate the training and testing data, find the MFCC features, perform PCA and save it in a pickle file with name ‘dataset_standarized_all_10.pickle’ for further processing.
-	TrainData_SVM.py - This file will train the data on svm model, test the data and print the accuracy score.

### Results 

- Accuracy on training: 99.9%
- Accuracy on testing: 60%




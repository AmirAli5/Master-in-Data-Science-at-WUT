The file *preprocessing.ipynb* is responsible for loading and preforming text cleaning and initial preprocessing of the data.
It then saves in the porquet format, for faster loading.

The files *cnn_bow.ipynb*, *cnn_tf-idf.ipynb*, and *lstm.ipynb* are responsible for the building, training, and testing process
of the CNN trained on Bag of Words, CNN trained of TF-IDF, and the LSTM network respectively.

*lstm_test.ipynb* is initially a copy of *lstm.ipynb*, at the end it shows how the provided models can be loaded and tested manually.

*logs* directory contains logs of the training process

*figs* directory contains graphs of the training process

*models* directory contains the trained models
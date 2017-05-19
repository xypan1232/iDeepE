#DeepE: Inferring RNA-protein binding sites and motifs using local and global convolutional neural network
 

Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and compress it in current dir.

Now it supports GPUs and 3 models, including CNNs, CNN-LSTM and ResNet, they can be trained using local CNNs and global CNNs, and alos ensembling of local and global CNNs.

#Usage:
ideepe.py [-h] [--posi <postive_sequecne_file>]
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE]
                 [--out_file OUT_FILE] [--train TRAIN]
                 [--model_file MODEL_FILE] [--predict PREDICT]
                 [--testfile TESTFILE] [--maxsize MAXSIZE] [--channel CHANNEL]
                 [--window_size WINDOW_SIZE] [--local LOCAL] [--glob GLOB]
                 [--ensemble ENSEMBLE] [--batch_size BATCH_SIZE]
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS]


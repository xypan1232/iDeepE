# iDeepE: Inferring RNA-protein binding sites and motifs using local and global convolutional neural network
 
# Dependency:
PyTorch 0.1.11 (http://pytorch.org/ ) <br>
Sklearn (https://github.com/scikit-learn/scikit-learn)


# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and compress it in current dir. It has 24 experiments of 21 RBPs, and we need train one model per experiment.

Now it supports GPUs and 3 models, including CNNs, CNN-LSTM and ResNet, they can be trained using local CNNs and global CNNs, and alos ensembling of local and global CNNs.

#Usage:
python ideepe.py [-h] [--posi <postive_sequecne_file>] <br>
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE] <br>
                 [--out_file OUT_FILE] [--train TRAIN] <br>
                 [--model_file MODEL_FILE] [--predict PREDICT] <br>
                 [--testfile TESTFILE] [--maxsize MAXSIZE] [--channel CHANNEL] <br>
                 [--window_size WINDOW_SIZE] [--local LOCAL] [--glob GLOB] <br>
                 [--ensemble ENSEMBLE] [--batch_size BATCH_SIZE] <br>
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS] <br>
It supports model training, testing and different model structure, MODEL_TYPE can be CNNs, CNN-LSTM and ResNet.

# Use case:
You want to predict the binding sites fro RBP ALKBH5 using ensebling local and global CNNs, and the default model is ensembling model. <br>
You first need train the model, then the trained models is used to predict for your sequences, the follwoing CLI will train a ensembling model using local and global CNNs using training positves and negatives. <br>
# step 1:
1. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --train=True 
<br>
For ensembling models, it will save 'model.pkl.local' and 'model.pkl.global' for local and global CNNs, respectively.<br>

# step 2:
2. python ideepe.py --testfile=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa --model_type=CNN --model_file=model.pkl --predict=True 
<br>

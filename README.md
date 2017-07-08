# iDeepE: Inferring RNA-protein binding sites and motifs using local and global convolutional neural network 
<br>
Computational algorithms for identifying RNAs that bind to specific RBPs are urgently needed, and they can complement high-cost experimental  methods. Previous methods all focus on using entire sequences for model training, and local sequence information is completely ignored. On the other hand, local sequences provide genomic context recognized by RBPs. In this study, we develop a convolutional neural network (CNN) based method called iDeepE  to predict RBP binding sites and motifs using local and global sequences. 
 <br> <br>
 
# Dependency:
python 2.7 <br>
PyTorch 0.1.11 (http://pytorch.org/ ) <br>
Sklearn (https://github.com/scikit-learn/scikit-learn)


# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and compress it in current dir. It has 24 experiments of 21 RBPs, and we need train one model per experiment.
Anohter dataset is from https://github.com/gianlucacorrado/RNAcommender/tree/master/examples, 47 RBPs are used in this study.

# Supported models
Now it supports GPUs and 3 types of models, including CNN, CNN-LSTM and ResNet. Each model can be trained using local CNNs and global CNNs, and also ensembling of local and global CNNs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs.

# Usage:
python ideepe.py [-h] [--posi <postive_sequecne_file>] <br>
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE] <br>
                 [--out_file OUT_FILE] [--motif MOTIF] [--train TRAIN] <br>
                 [--model_file MODEL_FILE] [--predict PREDICT] [--motif_dir MOTIF_DIR]<br>
                 [--testfile TESTFILE] [--maxsize MAXSIZE] [--channel CHANNEL] <br>
                 [--window_size WINDOW_SIZE] [--local LOCAL] [--glob GLOB] <br>
                 [--ensemble ENSEMBLE] [--batch_size BATCH_SIZE] <br>
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS] <br>
It supports model training, testing and different model structure, MODEL_TYPE can be CNN, CNN-LSTM and ResNet.

# Use case:
You want to predict the binding sites for RBP ALKBH5 using ensebling local and global CNNs, and the default model is ensembling model. <br>
You first need train the model per RBP, then the trained model is used to predict binding probability of this RBP for your sequences, the follwoing CLI will train a ensembling model using local and global CNNs, which are trained using positves and negatives derived from CLIP-seq. <br>
# step 1:
1. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --train=True 
<br>
For ensembling models, it will save 'model.pkl.local' and 'model.pkl.global' for local and global CNNs, respectively.<br>

# step 2:
2. python ideepe.py --testfile=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa --model_type=CNN --model_file=model.pkl --predict=True 
<br>

testfile is your input fasta sequences file, and the predicted outputs for all sequences will be defaulted saved in "prediction.txt". Each value seperated by TAB corresponds to the preobability of being RBP binding site for the sequence in fasta file

# Identify motifs:
You need install WebLogo (http://weblogo.berkeley.edu/) and TOMTOM in MEME Suite(http://meme-suite.org/doc/download.html?man_type=web) to search identifyed motifs against known motifs of RBPs. And also you need has positive and negative sequences when using motif option. <br> 
<br>
# step 3:
3. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --motif=True --motif_dir=motifs

The identified motifs (PWMs, and Weblogo) are saved to be defaulted dir motifs (you can also use --motif_dir to configure your dir for motifs), and also include the report from TOMTOM.

# Contact
Xiaoyong Pan: xypan172436atgmail.com

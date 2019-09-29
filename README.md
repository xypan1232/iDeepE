# iDeepE: Inferring RNA-protein binding sites and motifs using local and global convolutional neural network 
<br>
Computational algorithms for identifying RNAs that bind to specific RBPs are urgently needed, and they can complement high-cost experimental  methods. Previous methods all focus on using entire sequences for model training, and local sequence information is completely ignored. On the other hand, local sequences provide genomic context recognized by RBPs. In this study, we develop a convolutional neural network (CNN) based method called iDeepE  to predict RBP binding sites and motifs using local and global sequences. For global CNNs, one of their drawback is their poor scalability with increasing sequence length. However, local CNNs break the entire seuqence to fixed-size subsequences, which can handle any long sequence 
 <br> <br>
 
# Dependency:
python 2.7 <br>
PyTorch 0.1.11 (http://pytorch.org/ ) : you just need change loss_list.append(loss.data[0]) to loss_list.append(loss.item()) in the code for pytorch v0.4 or above.<br>
Sklearn (https://github.com/scikit-learn/scikit-learn)


# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and decompress it in current dir. It has 24 experiments of 21 RBPs, and we need train one model per experiment.
Another dataset is from https://github.com/gianlucacorrado/RNAcommender/tree/master/examples, 47 RBPs with over 2000 binding sites are used in this study.

# Supported models
Now it supports GPUs and 4 types of models, including CNN, CNN-LSTM, DenseNet and ResNet. Each model can be trained using local CNNs and global CNNs, and also ensembling of local and global CNNs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs. In addition, iDeepE can also be adapted to protein binding sites on DNAs and identify DNA binding speciticity of proteins. 

# Usage:
python ideepe.py [-h] [--posi <postive_sequecne_file>] <br>
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE] <br>
                 [--out_file OUT_FILE] [--motif MOTIF] [--train TRAIN] <br>
                 [--model_file MODEL_FILE] [--predict PREDICT] [--motif_dir MOTIF_DIR]<br>
                 [--testfile TESTFILE] [--maxsize MAXSIZE] [--channel CHANNEL] <br>
                 [--window_size WINDOW_SIZE] [--local LOCAL] [--glob GLOB] <br>
                 [--ensemble ENSEMBLE] [--batch_size BATCH_SIZE] <br>
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS] <br>
It supports model training, testing and different model structure, MODEL_TYPE can be CNN, CNN-LSTM and ResNet, DenseNet.

# Use case:
Take ALKBH5 as an example, if you want to predict the binding sites for RBP ALKBH5 using ensembling local and global CNNs, and the default model is ensembling model. <br>
You first need train the model for RBP ALKBH5, then the trained model is used to predict binding probability of this RBP for your sequences. The follwoing CLI will train a ensembling model using local and global CNNs, which are trained using positves and negatives derived from CLIP-seq. <br>
# step 1:
1. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --train=True 
<br>
For ensembling models, it will save 'model.pkl.local' and 'model.pkl.global' for local and global CNNs, respectively.<br>

# step 2:
2. python ideepe.py --testfile=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa --model_type=CNN --model_file=model.pkl --predict=True 
<br>

testfile is your input fasta sequences file, and the predicted outputs for all sequences will be defaulted saved in "prediction.txt". The value in each line corresponds to the probability of being RBP binding site for the sequence in fasta file. NOTE:if you have positive and negative sequecnes, please put them in the same sequecne file, which is fed into model for prediciton. DO NOT predict probability for positive and negative sequence seperately in two fasta files, then combine the prediction.

# Identify motifs:
You need install WebLogo (http://weblogo.berkeley.edu/) and TOMTOM in MEME Suite(http://meme-suite.org/doc/download.html?man_type=web) to search identifyed motifs against known motifs of RBPs. And also you need has positive and negative sequences when using motif option. <br> 
<br>
# step 3:
3. python ideepe.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --model_file=model.pkl --motif=True --motif_dir=motifs

The identified motifs (PWMs, and Weblogo) are saved to be defaulted dir motifs (you can also use --motif_dir to configure your dir for motifs), and also include the report from TOMTOM.

# NOTE
When you train iDeepE on your own constructed benchmark dataset, if the training loss cannot converge, may other optimization methods, like SGD or RMSprop can be used to replace Adam in the code. 

# Contact
Xiaoyong Pan: xypan172436atgmail.com

# Reference
 <p><b>Xiaoyong Pan^</b>, Hong-Bin Shen^. <a href="https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty364/4990826">Predicting RNA-protein binding sites and motifs through combining local and global deep convolutional neural networks</a>. Bioinformatics. In press. </p>

# Updates:
7/27/2017: add support network for DenseNet https://arxiv.org/abs/1608.06993, and fix the bug when generating binding motifs, and update the identified motifs for RBPs in GraphProt dataset.



# TransOMCS


This is the github repo for IJCAI 2020 paper "TransOMCS: From Linguistic Graphs to Commonsense Knowledge".

If you only want to use TransOMCS, you can **download** it from [TransOMCS](https://hkustconnect-my.sharepoint.com/:t:/g/personal/hzhangal_connect_ust_hk/Edq87bbgMXFInEJFbkNXq2kBwuC9jZM5ojlL5uaY8Ytu-g?e=zVoymh).

If you want to repeat the process of creating TransOMCS with OMCS and ASER, please follow the following steps.

## Dependency

Python 3.6, Pytorch 1.0

## Preparation


1. Download the core version of ASER from [ASER Homepage](https://hkust-knowcomp.github.io/ASER/) and install ASER 0.1 following [the guideline](https://github.com/HKUST-KnowComp/ASER/blob/master/ASER.ipynb).
2. Download the selected Commonsense OMCS Tuples and associated ASER graphs from [OMCS and ASER matches](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/EfFZFamzsmdKozyrU0-TtXsBDbStkt_FmPyeFM2kT-K9FQ?e=noAb7u).
3. Download the randomly split knowledge ranking dataset from [Ranking Dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/Efc7NeRYSVpHqcGuflDU3uoBRPaks4Mz1kG_R9OUwviPLw?e=oJB3yA).

## Construction of TransOMCS


(1) Unzip the downloaded matched OMCS tuple and ASER graphs in the same folder.

(2) Extract patterns: `python Pattern_Extraction.py`.

(3) Apply the extracted patterns to extract knowledge from ASER (You need to modify the location of your .db file): `python Knowledge_Extraction.py`.

(4) Train a ranking model to rank extracted knowledge: `python Train_and_Predict.py`.


## Application of TransOMCS


#### Reading Comprehension
Please use the code in [reading comprehension model](https://github.com/intfloat/commonsense-rc) and replace the external knowledge with different subsets of TransOMCS based on your need.

#### Dialog Generation
Please use the code in [dialog model](https://github.com/HKUST-KnowComp/ASER/tree/master/experiment/Dialogue) and replace the external knowledge with different subsets of TransOMCS based on your need.



## Others
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](mailto:hzhangal@cse.ust.hk), I will respond to that as soon as possible.
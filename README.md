# TransOMCS


This is the github repo for IJCAI 2020 paper ["TransOMCS: From Linguistic Graphs to Commonsense Knowledge"](https://arxiv.org/abs/2005.00206).

## Dependency

Python 3.6, Pytorch 1.0


## Introduction of TransOMCS

If you only want to use TransOMCS, you can **download** it from [TransOMCS](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/EVeNd_qvealEiTi7gs0Xu6sBbPIZI5ncD7Z1MBMdOz5CXw?e=E2Rtn0).

Without any further filtering, TransOMCS contains 20 commonsense relations, 101 thousand unique words, and 18.48 million triplets.

Here are the statistics and examples of different commonsense relations.

| Relation Name | Number of triplets | Reasonable Ratio | Example|
| :---: | :---: | :---: | :---:|
| CapableOf | 6,145,829 | 58.4% | (government, CapableOf, protect) |
| UsedFor | 3,475,254 | 50.8% | (kitchen, UsedFor, eat in) |
| HasProperty | 2,127,824 | 59.1% | (account, HasProperty, established) |
| AtLocation | 1,969,298 | 51.3% | (dryer, AtLocation, dishwasher) |
| HasA | 1,562,961 | 68.9% | (forest, HasA, pool) |
| ReceivesAction | 1,492,915 | 53.7% | (news, ReceivesAction, misattribute) |
| InstanceOf | 777,688 | 52.2% | (atlanta, InstanceOf, city) |
| PartOf | 357,486 | 62.8% | (player, PartOf, team) |
| CausesDesire | 249,755 | 52.0% | (music, CausesDesire, listen) |
| MadeOf | 114,111 | 55.3% | (world, MadeOf, country) |
| CreatedBy | 52,957 | 64.6% | (film, CreatedBy, director) |
| Causes | 50,439 | 53.4% | (misinterpret, Causes, apologize) |
| HasPrerequisite | 43,141 | 62.7% | (doubt, HasPrerequisite, bring proof) |
| HasSubevent | 18,904 | 56.1% | (be sure, HasSubevent, ask) |
| MotivatedByGoal | 15,322 | 55.8% | (come, MotivatedByGoal, fun) |
| HasLastSubevent | 14,048 | 58.9% | (hungry, HasLastSubevent, eat) |
| Desires | 10,668 | 56.4% | (dog, Desires, play) |
| HasFirstSubevent | 2,962 | 58.4% | (talk to, HasFirstSubevent, call) |
| DefinedAs | 36 | 37.5% | (door, DefinedAs, entrance) |
| LocatedNear | 19 | 85.7% | (shoe, LocatedNear, foot) |

The reasonable ratio scores are annotated on the random sample over all of the extracted knowledge (no knowledge ranking). 

In general, TransOMCS is still quite noisy because TransOMCS is extracted from raw data with patterns. 
However, as shown in the paper, a careful use of the data in the downstream applications helps.
We will keep working on improving its quality.

## Construction of TransOMCS

If you want to repeat the process of creating TransOMCS with OMCS and ASER, please follow the following steps.

1. Download the core version of ASER from [ASER Homepage](https://hkust-knowcomp.github.io/ASER/) and install ASER 0.1 following [the guideline](https://github.com/HKUST-KnowComp/ASER/blob/master/ASER.ipynb).
2. Download the selected Commonsense OMCS Tuples and associated ASER graphs from [OMCS and ASER matches](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/EfFZFamzsmdKozyrU0-TtXsBDbStkt_FmPyeFM2kT-K9FQ?e=noAb7u).
3. Download the randomly split knowledge ranking dataset from [Ranking Dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/Efc7NeRYSVpHqcGuflDU3uoBRPaks4Mz1kG_R9OUwviPLw?e=oJB3yA).
4. Unzip the downloaded matched OMCS tuple and ASER graphs in the same folder.
5. Extract patterns: `python Pattern_Extraction.py`.
6. Apply the extracted patterns to extract knowledge from ASER (You need to modify the location of your .db file): `python Knowledge_Extraction.py`.
7. Train a ranking model to rank extracted knowledge: `python Train_and_Predict.py`.


## Application of TransOMCS


#### Reading Comprehension
Please use the code in [reading comprehension model](https://github.com/intfloat/commonsense-rc) and replace the external knowledge with different subsets of TransOMCS based on your need.

#### Dialog Generation
Please use the code in [dialog model](https://github.com/HKUST-KnowComp/ASER/tree/master/experiment/Dialogue) and replace the external knowledge with different subsets of TransOMCS based on your need.

## TODO

1. Filter the current TransOMCS to further improve the quality (e.g., merge pronouns like 'he' and 'she' to human).

## Citation

    @inproceedings{zhang2020TransOMCS,
      author    = {Hongming Zhang and Daniel Khashabi and Yangqiu Song and Dan Roth},
      title     = {TransOMCS: From Linguistic Graphs to Commonsense Knowledge},
      booktitle = {Proceedings of International Joint Conference on Artificial Intelligence (IJCAI) 2020},
      year      = {2020}
    }

## Others
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](mailto:hzhangal@cse.ust.hk), I will respond to that as soon as possible.
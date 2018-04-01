# Learning on Partial-Order Hypergraphs
Fuli Feng, Xiangnan He, Yiqun Liu, Liqiang Nie, and Tat-Seng Chua (2018). Learning on Partial-Order Hypergraphs. In Proceedings of the 26th International Conference on World Wide Web, Lyon, France, April 23-27, 2018

We are finding license suitable to release this software. Codes will be released later.

Last update Date: 1 Apr., 2018
<!---Source code of partial-order hypergraph and baselines in our WWW 2018 paper, Learning on Partial-Order Hypergraphs

## Requirements

* TensorFlow (1.2)
* Python (2.7)

## Data

You need to download the feature file of micro-videos and save it under path (data/mvp/) before run the scripts about micro-video popularity prediction.

## University Ranking

Change working directory

```
cd cur
```

Run scripts

| Method | Command |
| :-----------: | :-----------: |
| **Simple_Graph** | ```python graph_ranking.py``` |
| **Hypergraph** | ```python graph_ranking.py -t hyper``` |
| **POH_Salary** | ```python poh_ranking.py``` |
| **POH_NCEE** | ```python poh_ranking.py -t ncee``` |
| **POH_All** | ```python poh_ranking.py -t all``` |
--->
## Micro-Video Popularity Prediction

Download video features: [feature.csv.tgz](https://drive.google.com/open?id=1A8PuXN9CFTJpocAHq0kPa0vbfYZ95ZRh) and replace the feature.csv under data/mvp.

<!---Change working directory

```
cd mvp
```

Run scripts

| Method | Command |
| :-----------: | :-----------: |
| **Simple_Graph** | ```python graph_regression.py``` |
| **Hypergraph** | ```python graph_regression.py -t hyper``` |
| **POH_Follow** | ```python poh_regression.py -t follow``` |
| **POH_Loop** | ```python poh_regression.py -t loop``` |
| **POH_All** | ```python poh_regression.py -t all``` |
| **GCN** | ```python gcn_regression.py``` |
| **LR_HG** | ```python lr_hg_regression.py``` |
| **LR_POH** | ```python lr_poh_regression.py``` |

## Cite

If you use the code, please kindly cite the following paper:
```
@inproceedings{fuli2018learning,
  title={Learning on Partial-Orde Hypergraphs},
  author={Feng, Fuli and He, Xiangnan and Liu, Yiqun, and Nie, Liqiang and Chua Tat-Seng},
  booktitle={Proceedings of the 26th International Conference on World Wide Web},
  year={2018},
  organization={ACM}
}
```

## Contact

fulifeng93@gmail.com
--->

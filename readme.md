### SubKNet

Code and datasets for "Revealing Herb-Symptom Associations and Mechanisms of Action in Protein Networks Using Subgraph Matching Learning"

**Developers**

Menglu Li ([mengluli@foxmail.com](mailto:mengluli@foxmail.com)) , Yongkang Wang (wyky481@webmail.hzau.edu.cn), Yujing Ni, and Wen Zhang ([zhangwen@mail.hzau.edu.cn](mailto:zhangwen@mail.hzau.edu.cn)) from College of Informatics, Huazhong Agricultural University.

**Tuturial**

To get the experiment results by running the scripts (scenario ):

```
python main_cv.py --dataset MDA --save_dir results_cv_{scenario}/ --data_dir ../data/{scenario}/   ###5-fold cross-validation
```

```
python main_indep.py --dataset MDA --save_dir results_indep_{scenario}/ --data_dir ../data/{scenario}/   ###independent test
```

Users can use their **own data** to train prediction models.

```
python save_train_val_test_data.py   ###splitting the data by different strategies
```

**Requirements**

- numpy 1.25.0
- pandas 1.5.3
- networkx 3.1
- scipy 1.10.1
- scikit-learn 1.2.2
- pytorch 2.0.1
- torch-geometric 2.3.1
- python 3.9.18

**Contact**

Please feel free to contact us if you need any help.
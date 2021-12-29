# ml4hep

info: https://agenda.infn.it/event/28573/

This is a 2 days Training Course on Machine Learning for beginners focussed on High Energy Physics applications. 

The course consists in theory (slides can be found [here](https://github.com/cfteach/ml4hep/blob/main/slides/ML4HEP_1%2B2.pdf)) and hands-on sessions and covers the following topics:

Day 1
- `Bias/Variance in Machine Learning` 
- `Gradient Descent` (hands-on [OGD, SGD, NAG, ADAM, RMSProp](https://github.com/cfteach/ml4hep/blob/main/gradient/gradient_descent.ipynb))
- `Linear and Logistic Regression` 
- `Combination of Models (Ensembles, Bagging, Boosting, Random Forests, GBT, XGBoost)`  (hands-on [XGBoost Ex. 1](https://github.com/cfteach/ml4hep/blob/main/xgboost/XGBoost_SUSY.ipynb), [Ex. 2](https://github.com/cfteach/ml4hep/blob/main/xgboost/XGBoost_higgs_v2.ipynb))
- `Clustering (K-Means, Density-based clustering methods: DBSCAN, HDBSCAN)` (hands-on [Clustering](https://github.com/cfteach/ml4hep/blob/main/clustering/clustering.ipynb)) 

Day 2:
- `Introduction to Neural Networks and hyperparameter optimization` (hands-on [how to build your first feed-forward NN with PyTorch](https://github.com/cfteach/ml4hep/blob/main/dnn/DNN_SUSY_gpu.ipynb))
- `Detector Design Optimization: single and multi-objective optimization` (hands-on [Bayesian Optimization](https://github.com/cfteach/ml4hep/blob/main/design_optimization/driver_bo.ipynb), [Multi-objective Optimization with meta-heuristic Ex. 2](https://github.com/cfteach/ml4hep/blob/main/design_optimization/driver_moo_sol2.ipynb), [MOO Ex. 3 (generalization to 3 objectives)](https://github.com/cfteach/ml4hep/blob/main/design_optimization/driver_moo_3obj_sol3.ipynb)) 


## Credits/References: 

The course utilizes the following as main reference: 

[1] `A high bias, low-variance introduction to Machine Learning` [hblvi2ML]: https://arxiv.org/abs/1803.08823

Other references:

[2] `Deep Learning`, Ian Goodfellow, Yoshua Bengio and Aaron Courville, https://www.deeplearningbook.org

[3] `Information Theory, Inference, and Learning Algorithms`, David J.C. MacKay, https://www.inference.org.uk/itprnn/book.pdf

[4]  AI4NP winter school, `Detector design optimization`, Cristiano Fanelli, https://github.com/cfteach/AI4NP_detector_opt 



## Requirements

python3; all other packages are installed from scratch during the course.

Documentation on `scikit-learn` can be found [here](https://scikit-learn.org/stable/).

Documentation on `PyTorch` can be found [here](https://pytorch.org).

We make use of [jupyter notebook](https://jupyter.org) and [colab](https://colab.research.google.com/). 


# GradientBoosting

My own *custom* implementation of [Gradient Boosting](http://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/gradient_boosting.pdf) based on my own *custom* implementation of [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree). On the average quality of this gradient gives in **2-4% higher** than sklearn's implementation.

The reason for the quality increasing:
* Using [Random Subspace Method](https://en.wikipedia.org/wiki/Random_subspace_method)
* **Adaptive** Desition Tree depth.

Moreover, the model allows you to use your own: 
* Loss Function (must be inherited from Abstract class). MSE and MAE are built-in.
* Custom **node purity** for decision tree (must be inherited from Abstract class). Gini and Entropy are built-in.

File Boosting.ipynb shows that this algorithm **works better than Sklearn's** on the given datasets with the same parameters.

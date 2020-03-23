# stats1952_midterm
Project investigating subgraphs revealed by filter pruning

The general pipeline for running scripts to generate the feather dataframe that is fed into the R portioned of this analysis is as follows:
1. Run simple_train.py to train a model (such as that stored in the models folder) from which Pruning Importance Metric will be drawn.
2. Run rank_by_class/submit_single_class.py to iteratively submit scripts that generate a filter ranking for each label class
3. Run make_rank_dataframe.py to generate the single feather object of all class ranks


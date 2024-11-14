# VERONA: predicti<i>VE</i> p<i>R</i>ocess m<i>O</i>nitoring be<i>N</i>chm<i>A</i>rk

**Version:** v1.1.1 (November 2024)

**Authors:**
 - **Efrén Rama-Maneiro**: [GitLab Profile](https://gitlab.citius.usc.es/efren.rama) -
[Personal Page](https://citius.gal/team/efren-rama-maneiro)
 - **Pedro Gamallo-Fernández**: [GitLab Profile](https://gitlab.citius.usc.es/pedro.gamallo) -
[Personal Page](https://citius.gal/team/pedro-gamallo-fernandez)


The ***VERONA*** library is a powerful Python tool designed to evaluate and compare predictive process
monitoring models *fairly and under equals conditions*. Leveraging the benchmark published in Rama-Maneiro et al. [1],
this library provides comprehensive functions for assessing the performance of predictive models in the context of 
business process monitoring.

## Key Features
- **Benchmark-Based Metrics:** Utilize established benchmarks from the aforementioned paper to evaluate your 
predictive models, ensuring *fair and standardized comparisons*.
- **Model Comparison:** Easily compare different predictive process monitoring models using a variety of metrics 
tailored to business process contexts.
- **Dataset Splitting:** Implement efficient dataset splitting techniques, including hold-out and cross-validation 
schemes, to facilitate rigorous testing and validation of your models.
- **User-Friendly Interface:** Intuitive functions and clear documentation make it easy for users to integrate the 
library into their projects and research workflows.

## Instalation
You can install the library from this repository as follows:
- Create a virtual environment (preferably a Conda environment):
    ```bash
    conda create -n verona_env python==3.11
    ```
- Initialize the environment:
    ```bash
    conda activate verona_env
    ```
- Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- Install the code as library:
    ```bash
    python setup.py install
    ```

Or you can just install it using pip:
```bash
pip install verona
```

## Usage

- Download a dataset:
    ```python
    from verona.data import download

    # Get BPI 2012 Complete event log in csv
    _, dataset = download.get_dataset('bpi2012comp', store_path='./data/csv', extension='csv')
  
    # Get Helpdesk event log in xes
    _, dataset = download.get_dataset('helpdesk', store_path='./data/xes', extension='xes')
    ```

- Split data in partitions:
    ```python
    from verona.data import split
  
    # Split data using 5-fold cross-validation scheme
    train_dfs, val_dfs, test_dfs = split.make_crossvalidation(dataset, store_path='./data/csv/cv', cv_folds=5,
                                                              val_from_train=0.2, case_column='case:concept:name')
  
    # Split data using hold-out scheme
    train_df, val_df, test_df = split.make_holdout(dataset, store_path='./data/csv/holdout', test_size=0.2,
                                                   val_from_train=0.2, case_column='case:column:name')                                                      
    ```
  
- Extract prefixes and target:
    ```python
    from verona.data import extractor
  
    # Extract all the prefixes and its targets for activity suffix prediction
    train_prefixes, train_targets = extractor.get_prefixes_and_targets(train_df, 'activity_suffix',
                                                                       case_id='case:concept:name',
                                                                       activity_id='concept:name')
  
    # Extract 3-prefixes and its targets for next activity prediction 
    train_prefixes, train_targets = extractor.get_prefixes_and_targets(train_df, 'next_activity', prefix_size=3,
                                                                       case_id='case:concept:name',
                                                                       activity_id='concept:name')
    ```

- Obtain metrics results:
    ```python
    from verona.evaluation.metrics import event
    import numpy as np
  
    targets = np.array(list(targets.values()))
    # Calculate F1-Score
    f1_score = event.get_f1_score(predictions, targets, preds_format='onehot', gt_format='labels')
    ```

- Calculate statistical tests:
    ```python
    from verona.evaluation.stattests.plackettluce import  PlackettLuceRanking
    import pandas as pd
  
    result_matrix = pd.DataFrame([[0.75, 0.6, 0.8], [0.8, 0.7, 0.9], [0.9, 0.8, 0.7]])
    plackett_ranking = PlackettLuceRanking(result_matrix, ["a1", "a2", "a3"])
    results = plackett_ranking.run(n_chains=10, num_samples=300000, mode="max")
    ```
- Plot the results
    ```python
    from verona.visualization import metrics
    import pandas as pd
    import numpy as np
  
    data = pd.DataFrame({
        'Helpdesk': np.array([80.3, 81.4, 80.1, 79.9, 80.9]),
        'BPI 2012': np.array([74.0, 74.2, 73.8, 74.5, 73.5]),
        'BPI 2013': np.array([64.2, 60.6, 61.3, 65.9, 60.8])
    })

    plt = metrics.bar_plot_metric(data, x_label='Datasets', y_label='Accuracies',
                                  reduction='mean', print_values=True)
    plt.show()
    ```

## License
This project is licensed under the terms of the **GNU General Public License v3.0** (*GPLv3*) - see the LICENSE file 
for details.

## References
1. Rama-Maneiro, E., Vidal, J., & Lama, M. (2021). Deep learning for predictive business process monitoring: Review and 
benchmark. IEEE Transactions on Services Computing.

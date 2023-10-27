# VERONA: predicti<i>V</i>e p<i>R</i>ocesss m<i>O</i>nitoring be<i>N</i>chm<i>A</i>rk

**Version:** v1.0.0 (October 2023)

**Authors:**
 - **Efrén Rama-Maneiro**: [GitLab Profile](https://gitlab.citius.usc.es/efren.rama) -
[Personal Page](https://citius.gal/team/efren-rama-maneiro)
 - **Pedro Gamallo-Fernández**: [GitLab Profile](https://gitlab.citius.usc.es/pedro.gamallo) -
[Personal Page](https://citius.gal/team/pedro-gamallo-fernandez)


The ***BARRO*** library is a powerful Python tool designed to evaluate and compare predictive process
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
conda create -n barro_env python==3.11
```
- Initialize the environment:
```bash
conda activate barro_env
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
```bash
import verona

# ToDo
```

## License
This project is licensed under the [ToDo] - see the LICENSE.txt file for details.

## References
1. Rama-Maneiro, E., Vidal, J., & Lama, M. (2021). Deep learning for predictive business process monitoring: Review and 
benchmark. IEEE Transactions on Services Computing.

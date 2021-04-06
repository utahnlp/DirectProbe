# DirectProbe

Codebase for probe representation space without training
classifiers.

See the [blog post on DirectProbe][blog] for a brief
introduction.

# Installing

1. Clone the repository

    ```
        git clone https://github.com/utahnlp/DirectProbe/
        cd DirectProbe
    ```

2. [Optional] Construct a virtual environment for this
   project. Only `python3` is supported.
    
    ```
        pyenv virtualenv 3.8.0 DirectProbe
        pyenv local DirectProbe
    ```

    More details about creating python virtual environment
    using `pyenv` can found [here][pyenv]

3. Install the required packages. `gurobipy` is
   installed independently because it is installed from a
   private PyPi server. Be note, here `gurobipy` is just a
   python interface for [Gurobi][]. You still need to
   install the REAL [Gurobi][] following the instrcutions
   from [Gurobi Installation Guide][] and get the licenses
   from [Gurobi][]. If you can not install the Gurobi or
   obtain the licenses, the probe will detect automatically and
   degrade to use linear SVM from [scikit-learn][]. However,
   using a linear SVM instead of [Gurobi][] results into unstable
   results. It might end up with different clusters and hard
   to reproduce.

    ```
        pip install -r requirements.txt
        pip install -i https://pypi.gurobi.com gurobipy
    ```

# Getting Started

## Download datasets and Running examples

1. Download the pre-packaged data from [here][data_url] and
   unzip them. Inside each dataset, there are three
   directories:
    
        - 'embeddings': contains all the embeddings from
          different representation models.
            - 'embeddings/layers': contains the embeddings
              from each layer of BERT-base-cased model.
        - 'entities': contains the examples of (example,
          label) pairs for training and test set. Example
          and label are separated by a tab. Each line is an
          example.
        - 'labels': contains the set of possible labels for
          each task.

2. Suppose all the pre-packaged data is put in the directory
   `data`, then we can run an experiment using the
   configuration from `config.ini`.

        sh run.sh

## Results

After probing, you will find the results in the
directory `results/SS/`.(We are using the supersense
role task as the example.)
In this directory, there are 4 files:

- `clusters.txt`: The clustering results. Each line contains
  a cluster number for the corresponding training example. 
- 'dis.txt': The distances between clusters. Each line
  represents a pair of cluters. The format is:

       (i-A,j-B): d

    where `i,j` is the cluster number, `A` and `B` are their
    corresponding label, `d` is the distance between these
    two clusters.
- 'log.txt': The probing log file.
- 'prediction.txt': The prediction results using the
  clusters. Each line is an example in the test set. Each
  line is in the following format:

        gold_label\t i-A,d_i\t j-B,d_j ...

    where `gold_label` is the golden label for the test
    example, `i` and `j` is the cluster number, 'A' and 'B'
    are the labels for corresponding cluster. 'd_i' is the
    distance between test point and cluster 'i'. All these
    clusters are sorted in increasing order.

# Configuration

DirectProbe probe the representations vis the configurations
from `config.ini` file. Please see the `config.ini` for more
details.


[blog]: http://research.zhouyichu.com/DirectProbe.html
[pyenv]: https://github.com/pyenv/pyenv
[Gurobi]: https://www.gurobi.com/
[Gurobi Installation Gurobi]: https://www.gurobi.com/documentation/9.1/quickstart_mac/software_installation_guid.html
[scikit-learn]: https://scikit-learn.org/stable/ 
[data_url]: https://drive.google.com/drive/folders/1cxYVXA6Oo2QoVowjRhBGqOqoRLUw6thq?usp=sharing

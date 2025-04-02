
# Q-Cluster: Quantum Error Mitigation Through Noise-Aware Unsupervised Learning

Q-Cluster provides tools and methods for quantum error mitigation utilizing noise-aware unsupervised learning algorithms.

---

## Installation

To install the necessary environment, use the following command:

```bash
conda env create -f environment.yml
```

Activate the environment after installation:

```bash
conda activate <your-environment-name>
```

Replace `<your-environment-name>` with the actual environment name defined in your `environment.yml` file.

---

## Usage

- **APIs**: All APIs related to **Q-Cluster** are implemented in [`src/qcluster.py`](src/qcluster.py).

- **Demonstration**: A small demonstration notebook [`src/demo.ipynb`](src/demo.ipynb) is provided to illustrate how to effectively use Q-Cluster.

- **Data**: Example data for running demonstrations on real quantum hardware can be found in the [`data/`](data/) directory.

---

## Project Structure

```
.
├── data/             # Data files for real machine demonstrations
├── src/
│   ├── qcluster.py   # Main APIs for Q-Cluster
│   └── demo.ipynb     # Demonstration notebook
├── environment.yml   # Environment dependencies
└── README.md         # Project description and instructions
```



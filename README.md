# C2TCFramework

The **C2TCFramework** is a simulation and optimization framework for resource allocation, network modeling, and environment visualization in communication systems. It provides tools for defining scenarios, simulating network behavior, and optimizing resource allocation for processes in complex environments.

## 🗂️ Project Structure

The project is organized into the following directories:

* **`configs/`**: Configuration files for scenarios, applications, and network topologies.

  * Example: `basescenario1.json`, `apps.json`
* **`datasets/`**: Data files for benchmarks, cloud models, and other datasets.

  * Example: `benchmark/`, `cloud/`
* **`deployments/`**: Deployment configurations for different environments.

  * Example: `env1.json`, `env2.json`
* **`environments/`**: Environment definitions, including dimensions and obstacles.

  * Example: `env1.json`, `env2.json`
* **`networking/`**: Modules for network modeling, including channel models and environment entities.

  * Example: `channel_model.py`, `environment.py`
* **`plotters/`**: Scripts for visualizing environments and network topologies.

  * Example: `draw_env.py`
* **`resourceallocation/`**: Core logic for resource allocation and optimization.

  * Example: `jnecora.py`, `compute_brcoverage_matrix.py`
* **`utils/`**: Utility functions for plotting, color management, and geometry.

  * Example: `plotting.py`, `colors.py`

## ✨ Key Features

* **Scenario Configuration**: Define environments, deployments, and network topologies using JSON files.
* **Resource Allocation**: Optimize resource allocation for processes using algorithms implemented in `jnecora.py`.
* **Network Modeling**: Simulate channel models and compute coverage matrices.
* **Visualization**: Plot environments, topologies, and allocation results using `matplotlib`.

## 💾 Installation

Clone the repository:

```bash
git clone https://github.com/marcopettorali/C2TCFramework
cd C2TCFramework
pip install -r requirements.txt
```

## 🧪 Usage

### ▶️ Running a Scenario

1. Define a scenario configuration in the `configs/` directory (e.g., `basescenario1.json`).
2. Run the resource allocation script:

```bash
python -m resourceallocation.jnecora basescenario1 --result-path results.json
```

### 🖼️ Visualizing Environments

Use the `plotters/draw_env.py` script:

```bash
python -m plotters.draw_env basescenario1 --show
```

### 🧭 Drawing Routes

Visualize a route between two nodes:

```bash
python -m resourceallocation.jnecora basescenario1 --draw-route src dest
```

## 🧾 Configuration Files

* **Environment Files** (`environments/`): Define the physical layout, including obstacles.
* **Deployment Files** (`deployments/`): Specify positions and radii of deployed nodes.
* **Scenario Files** (`configs/`): Combine environment, deployment, and application configurations.

A tutorial on how to create a scenario file is available in the `docs/` directory.

### 📘 Examples

Example Scenario: `basescenario1.json`

* **Environment**: `environments/env1.json`
* **Deployment**: `deployments/env1.json`
* **Applications**: `configs/apps.json`

Example Deployment (`env1.json`):

```json
[
  {"pos": [45, 30], "radius": 47},
  {"pos": [80, 70], "radius": 47}
]
```

## 📚 Citation

If you use this framework in your research, please cite:

```
@ARTICLE{10886960,
  author={Pettorali, Marco and Righetti, Francesca and Vallati, Carlo and Das, Sajal K. and Anastasi, Giuseppe},
  journal={IEEE Internet of Things Journal}, 
  title={J-NECORA: A Framework for Optimal Resource Allocation in Cloud-Edge-Things Continuum for Industrial Applications With Mobile Nodes}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JIOT.2025.3536700}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See the LICENSE file for full terms.

You may copy, distribute, and modify the software as long as you track changes/dates in source files and disclose your source code. Derivative works must also be licensed under the GPL.

## 📬 Contact

For questions or support, please contact the project maintainers.

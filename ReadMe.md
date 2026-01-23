# Algorithms for MCM Preparation

This repository was created during my personal preparation for the Mathematical Contest in Modeling (MCM).
I participated in the competition as a programmer, and this project serves as a long-term learning and experimentation codebase.

The repository contains:
- Algorithm implementations written in Python (`.py`)
- Algorithm explanations and derivations written in Jupyter Notebooks (`.ipynb`)
- Self-written extensions and visualization utilities

---

## Repository Structure

~~~text
algorithms/
├── <algorithm_name>/
│   ├── <algorithm_name>.py        # Core callable function(s)
│   ├── <algorithm_name>.ipynb     # Algorithm explanation and illustration
│   └── other_extensions.py        # Optional experimental or auxiliary code
~~~

- Each algorithm is organized in its own subdirectory.
- The `<algorithm_name>.py` file contains the primary function intended for direct reuse.
- Additional files are reserved for extensions, experiments, or visualization purposes.

---

## Implemented Algorithms

This project currently includes the following decision-making algorithms:

- **AHP (Analytic Hierarchy Process)**
- **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**
- **Entropy Weight Method**

Each algorithm is accompanied by implementation code and explanatory materials.

---

## Recent Updates

- Refactored the **TOPSIS** visualization module to improve configurability and control.
- Added clear inline comments to function interfaces for better readability and reuse.
- Implemented the **Entropy Weight Method**, including visualization support.

---

## Acknowledgements

This project is inspired by and partially based on the following open-source libraries:

- `pyDecision`
- `pyAHP`

Many implementations in this repository are adapted and extended from these projects for learning and experimentation purposes.

---

## Contribution

Issues and pull requests are welcome.

If this repository is helpful to you, feel free to give it a ⭐️.

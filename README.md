# yukawa-sindy

This repository contains code exploring the efficacy of the **Sparse Identification of Nonlinear Dynamical Systems (SINDy)** method to uncover governing equations of motion for **three-body interactions** in a **Yukawa potential**.  

The Yukawa potential serves as a simplified model for **interparticle forces in a dusty, low-temperature, unmagnetized plasma**, where charge screening leads to short-range interactions. This project uses **[PySINDy](https://pysindy.readthedocs.io/en/latest/)** to discover the underlying dynamical laws directly from simulated data, offering a data-driven approach to understanding nonlinear plasma dynamics.

---

## 🚀 Project Overview

The goal of this repository is to:
- Simulate the dynamics of three interacting particles under a **Yukawa potential**.
- Apply **SINDy** to the generated trajectory data with synthetic noise to **discover the governing equations**.
- Evaluate the accuracy, robustness, and generalizability of the learned models with different noise levels and **cross-validation**.
- Provide a **reproducible reference** for figures, data, and results used in a forthcoming scientific publication.

Note: This is primarily a **personal research repository**, serving as a record of the code and methods used to produce results and visualizations. If you have questions or concerns, feel free to submit an issue. 

---

# Usage Guide for DPC4PowerElectronics

This document provides a complete usage guide for the DPC4PowerElectronics toolbox.
It explains the role of each file in the repository and describes the recommended
workflow for data collection, neural network training, and closed-loop validation.

---

## 1. Repository Structure and File Roles

This repository contains MATLAB scripts, data files, and Simulink models.
Their purposes are summarized below.

---

### 1.1 Data Collection Model: `CCS_MPC.slx`

- This Simulink model is used to **collect training data** for the neural-network-based
  data-driven predictive controller (DPC).
- It implements a benchmark CCS/MPC control strategy.
- The model generates state, input, and reference trajectories required for training.

This model is **only required if new training data need to be generated**.

---

### 1.2 Training Data: `Data4train.mat`

- Contains **pre-generated training data**.
- Can be used directly without running `CCS_MPC.slx`.
- Recommended for users who want to quickly reproduce results or validate the method.

---

### 1.3 MATLAB Training Scripts (Root Directory Requirement)

The following MATLAB files must be placed in the **same root directory** of the repository:

- `Main.txt`  
- `Custom_Loss.txt`  
- `Model_gradients.txt`

These files are tightly coupled:

- `Main.txt` is the main training script.
- `Main.txt` internally calls the custom loss function defined in `Custom_Loss.txt`.
- `Model_gradients.txt` provides gradient computations for training.

> **Important:**  
> MATLAB requires these scripts to be accessible on the search path.
> Therefore, they must remain in the same root directory and should not be moved.

---

### 1.4 Validation Model: `DPCvalidation.slx`

- Simulink model for **closed-loop validation** of the trained DPC controller.
- Used to evaluate control performance and compare against benchmark controllers.
- This model **does not run directly** without proper initialization.

Before running this model, the trained neural network parameters must be
loaded into the MATLAB workspace.

---

## 2. Typical Workflow Overview

The recommended workflow consists of three stages:

1. (Optional) Data collection using `CCS_MPC.slx`
2. Neural network training using MATLAB scripts
3. Closed-loop validation using `DPCvalidation.slx`

Two usage options are provided depending on whether existing data are used
or new data are generated.

---

## 3. Option A: Use Existing Training Data (Recommended)

This option allows users to directly train and validate the controller
using the provided dataset.

### Step 1: Add Repository to MATLAB Path
```matlab
addpath(genpath(pwd))

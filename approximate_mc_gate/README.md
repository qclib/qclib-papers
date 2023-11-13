# Linear decomposition of approximate multi-controlled single qubit gates

---
The reference for this work is in [https://arxiv.org/abs/2310.14974](https://arxiv.org/abs/2310.14974).

---

## First Steps

### 1. Setting Up a Virtual Environment (Optional but Recommended)

Before installing dependencies, it's a best practice to create a virtual environment to isolate the project's libraries.
Run the following commands in the terminal:


```bash
# Install virtualenv (if not already installed)
pip install virtualenv

# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Linux/Mac)
source venv/bin/activate

# Activate the virtual environment (Windows)
.\venv\Scripts\activate
````
### 2. Installing Dependencies

With the virtual environment activated (if used), you can now install the dependencies from requirements.txt. 
Run the following command:

```bash
pip install -r requirements.txt
```
This command reads the requirements.txt file and installs all the listed libraries with their specified versions.

### 3. Starting the Jupyter Notebook

Now that all dependencies are installed, you can start the Jupyter Notebook:

```bash
jupyter notebook
```
This will open your default web browser in the Jupyter environment, where you can navigate to your notebook and open it.

Remember to keep your virtual environment activated whenever you are working on the project. When you're done, you can 
deactivate the virtual environment with the command:

```bash
deactivate
```

## List of figures:

In this section, we provide a list of key figures that have been generated for our paper. Each figure is referenced in 
the paper to enhance the reader's understanding. Below are the figures included:

- [Figure 6](#figure_6) - Comparison on the number of CNOTs between the method proposed Theorem 1 from this work against 
Lemma 7.8 from Ref. [8], for ϵ = 0.001.
- [Figure 10](#figure_10) - Analytical comparison of the number of CNOTs in the multi-controlled gate using multiple 
copies of the original SU(2) decomposition scheme.
- [Figure 11](#figure_11) - Comparison of the mathematical upper bound for the count of CNOTs of multi-controlled U(2) 
gates, for the approximated versions, nb = 13.
- [Figure 12](#figure_12) - Comparison of Quantum Circuit Decomposition CNOT cost for multi-controlled U = X gates 
- using different methods.

These graphics play a crucial role in conveying our results visually. For detailed insights, please refer to the 
corresponding sections in the paper where each figure is discussed in detail.

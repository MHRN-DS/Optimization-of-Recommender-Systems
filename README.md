# Frank-Wolfe: Optimization for Recommender Systems

This project explores the use of **Frank-Wolfe algorithms** for large-scale recommendation systems. Our approach focuses on solving convex optimization problems with sparsity constraints - in this specific case, the matrix completion problem -, leveraging variants of the Frank-Wolfe method including **Away-Step Frank-Wolfe**, **Pairwise Frank-Wolfe** and **In-Face Optimization**. Experiments were conducted on datasets such as **MovieLens 100k**, **Jester Jokes-1** and **MovieLens 1M**, demonstrating the algorithm's effectiveness for matrix completion tasks.

> **Grade received: 28.5/30**  
> **Contributors**: Max Hans-Jürgen Henry Horn, Mikhail Isakov (@Mishlen337), Lennart Niels Bredthauer (@Lenny945)  
> **University of Padova, Optimization for Data Science – July 2025**  
> Supervisor: Prof. Dr. Francesco Rinaldi

---

## Overview

The Frank-Wolfe algorithm (also known as Conditional Gradient) is well-suited for large-scale structured optimization, especially when projection steps are expensive. In this project, we apply:

- **Classical Frank-Wolfe**
- **In-Face Away-Step Frank-Wolfe**
- **In-Face Pairwise Frank-Wolfe**

... to build an efficient recommendation engine via matrix factorization techniques.

---

## Project Structure

```
.
├── algorithms.py                # Frank-Wolfe algorithm implementations
├── solver.py                    # Optimization solver using FW variants
├── exp_MovieLens_1e6.ipynb      # Experiment: MovieLens 1M dataset
├── exp_MovieLens_1e5.ipynb      # Experiment: MovieLens 100K subset
├── jester_joker_exp.ipynb       # Experiment: Jester joke dataset
├── Group_10_ODS_Recommender_Systems.pdf  # Final report (28.5/30)
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MHRN-DS/Optimization-of-Recommender-Systems.git
   cd Optimization-of-Recommender-Systems
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy
   ```

3. Run experiments via Jupyter:
   ```bash
   jupyter notebook exp_MovieLens_1e6.ipynb
   ```

---

## Datasets Used

- [MovieLens 100k](https://grouplens.org/datasets/movielens/)
- [Jester Joke Dataset](https://goldberg.berkeley.edu/jester-data/)
- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

---

## Key Features

- Efficient implementation of multiple Frank-Wolfe algorithm variants
- Scalable optimization for collaborative filtering
- Comparison between datasets and regularization techniques
- Highly modular structure for reuse in other convex optimization contexts

---

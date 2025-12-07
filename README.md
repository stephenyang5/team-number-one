# GTNode: Trajectory Inference with Graph Attention

**Angelic McPherson, Stephen Yang, Erica Shivers, and Hanting Li**  
Brown University, Department of Computer Science

## Overview
GTNode is our final project for *Deep Learning in Genomics*. It implements a graph-based deep learning framework that uses Graph Transformers and Neural ODEs to model dynamic cell-state transitions from single-cell RNA-seq data.

## Summary
GTNode constructs a weighted cell–cell graph from transcriptomic similarity and RNA velocity, applies a Graph Transformer to capture global structure, and integrates a Neural ODE module for continuous trajectory modeling and fate prediction. The goal is to provide a scalable, interpretable method for reconstructing developmental lineages and cellular differentiation pathways.


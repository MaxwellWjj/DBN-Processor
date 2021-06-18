# DBN-Processor
An energy-efficient processor design based on Deep Belief Network (DBN), which is one of the most suitable DNN models for on-chip learning

## Introduction
With the growing interest of edge computing in the Internet of Things (IoT), Deep Neural Network (DNN) hardware processors/accelerators face challenges of low energy consumption, low latency, and data privacy issues. This work proposes an energy-efficient processor design based on Deep Belief Network (DBN), which is one of the most suitable DNN models for on-chip learning. In this study, a thoroughly algorithm-architecture-circuit design optimization method is used for efficient design. The characteristics of data reuse and data sparsity in the DBN learning algorithm inspires this study to propose a heterogeneous multi-core architecture with local learning. In addition, novel circuits of transposable weight memory and sparse address generator are proposed to reduce weight memory access and exploit neuron state sparsity, respectively, for maximizing the energy efficiency. The DBN processor is implemented and thoroughly evaluated on Xilinx Zynq FPGA. Implementation results confirm that the proposed DBN processor has excellent energy efficiency of 45.0 pJ per neuron-weight update, which has been improved by 74% against the conventional design.

## Paper
The state-of-art paper of this work is still under review currently (submitted to IEEE JETCAS).  
The former version is a conference paper:  
`J. Wu et al., "An Energy-efficient Multi-core Restricted Boltzmann Machine Processor with On-chip Bio-plausible Learning and Reconfigurable Sparsity," 2020 IEEE Asian Solid-State Circuits Conference (A-SSCC), 2020, pp. 1-4, doi: 10.1109/A-SSCC48613.2020.9336135.`

## Open source
- ./rtl_src contains hardware design (system verilog). Please note that we only upload single RBM hardware here because the DBN is actually consists of several RBMs.
- ./py_src contains python script for simulation (Pytorch is needed in this project). Note that before using this simulation, end-users should generate MNIST dataset in .npy format.

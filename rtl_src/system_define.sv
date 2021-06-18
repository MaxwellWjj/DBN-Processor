//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: System definition (bit width...)
// Project Name: DBN Processor
// Target Devices: Zynq-7100
// Tool Versions: Vivado 2019.1
// Description: 1- All the parameters are defined in system_define.sv file, instead of using #(parameter ...).
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// 
//////////////////////////////////////////////////////////////////////////////////


// Parameters of transposed memory
`define NUM_TM_V 3
`define NUM_TM_H 3
`define NUM_WEIGHTS 9
`define BW_ADDR 2
`define DEPTH_ADDR_FIFO 8
`define AW_ADDR_FIFO 3

// Parameters of mapping
`define NUM_CORE_V 3
`define NUM_CORE_H 3
`define NUM_CORE 9
`define NUM_VN_ONECORE 3
`define NUM_HN_ONECORE 3
`define NUM_N_MAX_ONECORE 3
`define BW_CORE_INDEX 4

// Width of weights
`define BW_WEIGHTS_SIGN 1
`define BW_WEIGHTS_INT 0
`define BW_WEIGHTS_FRAC 8
`define BW_WEIGHTS (`BW_WEIGHTS_SIGN + `BW_WEIGHTS_INT + `BW_WEIGHTS_FRAC)

// Width of partial summation
`define BW_PS_SIGN 1
`define BW_PS_INT 2
`define BW_PS_FRAC 8
`define BW_PS (`BW_PS_SIGN + `BW_PS_INT + `BW_PS_FRAC)

// Width of total summation
`define BW_TS_SIGN 1
`define BW_TS_INT 4
`define BW_TS_FRAC 8
`define BW_TS (`BW_TS_SIGN + `BW_TS_INT + `BW_TS_FRAC)

// Width of gibbs sampling
`define BW_SAMP_INPUT `BW_TS

// Width of VPF learning rule (VPF only)
`define BW_DELTA_SIGN 1
`define BW_DELTA_INT 6
`define BW_DELTA_FRAC 8
`define BW_DELTA (`BW_DELTA_SIGN + `BW_DELTA_INT + `BW_DELTA_FRAC)
`define ITERATION_NUM 6

// Finite state machine
`define IDLE 3'b000
`define INIT 3'b001
`define GEN_VH 3'b010
`define REC_HV 3'b011
`define UPDATE 3'b100

// Additional states of addr_gen
`define ADDR_VH 3'b101
`define ADDR_HV 3'b111

// Learning rate (right shift)
`define LEARNING_RATE 8

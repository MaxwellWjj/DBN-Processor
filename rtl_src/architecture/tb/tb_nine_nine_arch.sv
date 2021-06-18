//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: testbench for 9 VNs and 9 HNs architecture
// Module Name: tb_nine_nine_arch
// Project Name: DBN Processor
// Target Devices: Zynq-7100
// Tool Versions: Vivado 2019.1
// Description: 1- This module needs faster clock to generate addresses.
//
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// 
//////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

`include "../../system_define.sv"

module tb_nine_nine_arch();
    
    logic clk;
    logic rst;
    logic en;
    logic [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] init_weight;
    logic [`BW_CORE_INDEX - 1 : 0] init_weight_index;
    logic init_w_en;

    logic [`NUM_VN_ONECORE * `NUM_CORE_V - 1 : 0] init_v;
    logic begin_operation;  // Single cycle signal, enabling the operation of RBM

    logic [`BW_CORE_INDEX - 1 : 0] out_index;
    wire [`NUM_HN_ONECORE - 1 : 0] infer_h;
    wire [`NUM_HN_ONECORE - 1 : 0] debug_h0_o;
    wire [`NUM_VN_ONECORE - 1 : 0] debug_v2_o;
    wire [`NUM_HN_ONECORE - 1 : 0] debug_h2_o;

    nine_nine_test_arch test_arch (clk, rst, en, init_weight, init_weight_index, init_w_en, init_v, begin_operation, out_index, 
                                    infer_h, debug_h0_o, debug_v2_o, debug_h2_o);

    always begin
        #5 clk <= ~clk;
    end 

    initial begin
        clk<=0;
        rst<=1'b1;
        en<=0;
        init_w_en <= 1'b0;
        begin_operation <= 1'b0;   
        init_weight<=27'h1002003;
        init_v <= 9'b101011101;
        init_weight_index <= 4'd0;
        out_index <= 0;
        // Initialization
        #20 rst<=0;
        #17 en<=1'b1;
        #20 init_w_en <= 1'b1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_weight_index <= init_weight_index + 1;
        #10 init_w_en <= 1'b0;
        // One training
        #40 begin_operation <= 1'b1;
        #10 begin_operation <= 1'b0;

    end


endmodule
//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: testbench for addr_gen and twm
// Module Name: tb_addrgen_twm
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

module tb_rbmcore();
    
    logic clk;
    logic rst;
    logic en;
    logic [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] init_weight;

    logic begin_operation;  // Single cycle signal, enabling the operation of RBM

    logic [`NUM_VN_ONECORE - 1 : 0] init_v;
    logic init_w_en;

    logic [`NUM_HN_ONECORE - 1 : 0] infer_h;

    logic [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] final_weight;

    logic done_vh;
    logic done_hv;
    logic receivevh;
    logic receivehv;

    logic signed [`BW_PS * `NUM_HN_ONECORE - 1 : 0] rbm_write_channel_data_vh;

    logic signed [`BW_PS * `NUM_VN_ONECORE - 1 : 0] rbm_write_channel_data_hv;


    logic [`NUM_VN_ONECORE - 1 : 0] new_states_v;
    logic [`NUM_VN_ONECORE - 1 : 0] new_states_v_en;
    logic [`NUM_HN_ONECORE - 1 : 0] new_states_h;
    logic [`NUM_HN_ONECORE - 1 : 0] new_states_h_en;

    logic [`NUM_HN_ONECORE - 1 : 0] debug_h0;
    logic [`NUM_VN_ONECORE - 1 : 0] debug_v2;
    logic [`NUM_HN_ONECORE - 1 : 0] debug_h2;

    rbm_core test_rbmcore (clk, rst, en, init_weight, 1'b0, 1'b0, begin_operation, init_v,
                            init_w_en, infer_h, final_weight, done_vh, done_hv, receivevh, receivehv,
                            rbm_write_channel_data_vh, rbm_write_channel_data_hv, new_states_v, new_states_v_en, 
                            new_states_h, new_states_h_en, debug_h0, debug_v2, debug_h2);

    always begin
        #5 clk <=~clk;
    end 

    initial begin
        clk<=0;
        rst<=1'b1;
        en<=0;
        init_w_en <= 1'b0;
        begin_operation <= 1'b0;
        receivevh <= 1'b0;
        receivehv <= 1'b0;        
        init_weight<=27'h1002003;
        init_v <= 3'b101;
        #20 rst<=0;
        #17 en<=1'b1;
        #20 init_w_en <= 1'b1;
        #10 init_w_en <= 1'b0;
        #100 begin_operation <= 1'b1;
        #10 begin_operation <= 1'b0;
        // V->H generation
        #150 receivevh <= 1'b1;
        #10 receivevh <= 1'b0;
        #50 new_states_h_en <= 3'b011;
        new_states_h <= 3'b010;
        #10 new_states_h_en <= 3'b000;
        new_states_h <= 3'b000;
        #10 new_states_h_en <= 3'b100;
        new_states_h <= 3'b100;
        #10 new_states_h_en <= 3'b000;
        new_states_h <= 3'b000;
        // H->V reconstruction
        #150 receivehv <= 1'b1;
        #10 receivehv <= 1'b0;        
        #50 new_states_v_en <= 3'b010;
        new_states_v <= 3'b010;
        #10 new_states_v_en <= 3'b101;
        new_states_v <= 3'b001;
        #10 new_states_v_en <= 3'b000;
        new_states_v <= 3'b000;
        // V->H generation
        #150 receivevh <= 1'b1;
        #10 receivevh <= 1'b0;        
        #50 new_states_h_en <= 3'b001;
        new_states_h <= 3'b000;
        #10 new_states_h_en <= 3'b000;
        #10 new_states_h_en <= 3'b100;
        new_states_h <= 3'b100;
        #10 new_states_h_en <= 3'b000;
        new_states_h <= 3'b000;
        #10 new_states_h_en <= 3'b010;
        new_states_h <= 3'b000;
        #10 new_states_h_en <= 3'b000;
    end


endmodule
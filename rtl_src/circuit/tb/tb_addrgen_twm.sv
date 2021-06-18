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


module tb_addrgen_twm();
    
    logic clk;
    logic rst;
    logic en;
    logic [2:0] statesignal;
    logic [1:0] haddr;
    logic [1:0] haddr_ff;
    logic [1:0] v_addr;
    logic [1:0] h_addr;
    logic [26:0] input_weights;
    logic [26:0] output_weights_vh;
    logic [26:0] output_weights_hv;
    logic [2:0] v_states;
    logic [2:0] h_states;
    logic en_fetch;
    logic [1:0] addr_fetch;
    logic empty_fifo;

    addr_gen t_addrgen (clk,rst,en,statesignal,v_states,h_states,en_fetch,addr_fetch,empty_fifo);
    t_mem t_memory (clk,rst,en,statesignal,v_addr,h_addr,input_weights,output_weights_vh,output_weights_hv);

    always begin
        #5 clk <=~clk;
    end 

    initial begin
        clk<=0;
        rst<=1;
        en<=0;
        statesignal <= 3'b000;
        v_states <= 3'b101;
        h_states <= 3'b110;
        haddr <= 2'b00;
        input_weights<=27'h1002003;
        #20 rst<=0;
        #10 en<=1;
        #10 statesignal <= 3'b001;
        #10 haddr <= 2'b01;
        #10 haddr <= 2'b10;
        #10 statesignal <= 3'b101;
        #30 statesignal <= 3'b010;
        #50 statesignal <= 3'b111;
        #30 statesignal <= 3'b011;
        #50 statesignal <= 3'b100;

    end

    assign en_fetch = ((statesignal == 3'b010 || statesignal == 3'b011) && (empty_fifo==0));
    assign h_addr = (statesignal == 3'b001)?haddr:(statesignal == 3'b011)?addr_fetch:0;
    assign v_addr = (statesignal == 3'b010)?addr_fetch:0;

endmodule
//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Sigmoid sampling function of ags cores
// Module Name: sampling
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

`include "system_define.sv"

`define THRESHOLD // THRESHOLD: using 0.5 as threshold; SIGMOID: using random number generator to sample the result

module sampling (
    // Clock, reset and enable
    input clk,
    input rst,
    input en,

    // Control and inout
    input wire signed [`BW_TS - 1 : 0] x,
    output logic state

);
    
    `ifdef SIGMOID
    logic [`BW_WEIGHTS_FRAC-1:0] lfsr_out;
    LFSR threshold_gen (.clk(clk), .rst(rst), .en(en), .out(lfsr_out));

    logic [`BW_WEIGHTS_FRAC - 1 : 0] sigmoid_1;
    logic [`BW_WEIGHTS_FRAC - 1 : 0] sigmoid_2;
    logic [`BW_WEIGHTS_FRAC - 1 : 0] sigmoid_3;
    logic [`BW_WEIGHTS_FRAC - 1 : 0] sigmoid_4;
    logic [`BW_WEIGHTS_FRAC - 1 : 0] sigmoid;
    logic [`BW_WEIGHTS_FRAC - 1 : 0] sampling_result;
    
    // TODO: modify them as parameterized statements.
    logic signed [`BW_TS - 1 : 0] x_abs;
    assign x_abs = x[`BW_TS - 1]? (~x + 1'b1) : x;            
    assign sigmoid_1 = 8'd255;
    assign sigmoid_2 = (x_abs >>> 5) + 8'd216;
    assign sigmoid_3 = (x_abs >>> 3) + 8'd160;
    assign sigmoid_4 = (x_abs >>> 2) + 8'd128;
    assign sigmoid = (x_abs>=13'd1280)?sigmoid_1:(x_abs>=13'd608)?sigmoid_2:(x_abs>=13'd256)?sigmoid_3:sigmoid_4;
    assign sampling_result = x[DW_TOTALSUM-1]?(8'd255-sigmoid):sigmoid;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= 1'b0;
        end
        else if (en) begin
            if(sampling_result >= lfsr_out) state<= 1'b1;
            else state <= 1'b0;
        end
    end
    `endif

    `ifdef THRESHOLD
    always_ff @( posedge clk ) begin
        if (rst) begin
            state <= 1'b0;
        end
        else if (en) begin
            if (~x[`BW_TS - 1]) begin
                state <= 1'b1;
            end
            else state <= 1'b0;
        end
    end
    `endif

endmodule
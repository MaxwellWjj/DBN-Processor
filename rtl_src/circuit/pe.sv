//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Processing element (PE), which is an accumulator of weights.
// Module Name: PE
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

module PE (
    // Clock, reset and enable
    input wire clk,
	input wire rst,	
	input wire en,

    // Control signal
    input wire [2:0] state_signal,

    // Input weights and output partial summation
    input wire signed [`BW_WEIGHTS - 1 : 0] weights,
    output logic signed [`BW_PS - 1 : 0] partial_sum
);

    always_ff @( posedge clk ) begin
        if (rst) begin
            partial_sum <= 0;
        end
        else if (state_signal == `GEN_VH || state_signal == `REC_HV) begin
            if (en) begin
                partial_sum <= partial_sum + weights;        
            end
        end
        else partial_sum <= 0;
    end
    
endmodule
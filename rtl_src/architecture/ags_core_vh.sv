//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: AGS Core
// Module Name: ags_core_vh
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

module ags_core_vh (
    // Clock, reset and enable
    input wire clk,
	input wire rst,
	input wire en,

    // From RBM core
    input wire data_in_en,
    input wire signed [`BW_PS - 1 : 0] data_in,

    // To RBM core
    output wire new_state,
    output logic new_state_en

);

    logic [4 : 0] cnt_acc; // Note that if the scale changes, the width should be modified.
    logic acc_finished;
    always_ff @( posedge clk ) begin
        if (rst) begin
            cnt_acc <= 0;
            acc_finished <= 1'b0;
        end
        else if (cnt_acc == `NUM_CORE_V) begin
            cnt_acc <= 0;
            acc_finished <= 1'b1;
        end
        else if (en & data_in_en) begin
            cnt_acc <= cnt_acc + 1;
            acc_finished <= 1'b0;            
        end
        else acc_finished <= 1'b0;
    end

    logic signed [`BW_TS - 1 : 0] total_sum; 
    always_ff @( posedge clk ) begin
        if (rst) begin
            total_sum <= 0;
        end
        else if (new_state_en) begin
            total_sum <= 0;
        end
        else if (en & data_in_en) begin
            total_sum <= total_sum + data_in;
        end
    end
    
    always_ff @( posedge clk ) begin
        if (rst) begin
            new_state_en <= 0;
        end
        else if (en) begin
            new_state_en <= acc_finished;
        end
    end
    
    sampling sample_unit (clk, rst, (en & acc_finished), total_sum, new_state);

endmodule
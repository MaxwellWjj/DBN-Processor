//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Arbiter circuit (basic)
// Module Name: arbiter_vh
// Project Name: DBN Processor
// Target Devices: Zynq-7100
// Tool Versions: Vivado 2019.1
// Description: 1- All the parameters are defined in system_define.sv file, instead of using #(parameter ...).
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - Note that this file should be carefully checked when the number of cores changes!
// 
//////////////////////////////////////////////////////////////////////////////////

`include "system_define.sv"

module arbiter_basic_hv (
    input [`NUM_CORE_H - 1 : 0] signal_in,
    output logic [`NUM_CORE_H - 1 : 0] arbiter_signal
);

    // The comb module should be modified if the number of cores changes
    always_comb begin
        casex(signal_in)
        6'b000000: arbiter_signal = 6'b000000;
        6'b000001: arbiter_signal = 6'b000001;
        6'b00001x: arbiter_signal = 6'b000010;
        6'b0001xx: arbiter_signal = 6'b000100;
        6'b001xxx: arbiter_signal = 6'b001000;
        6'b01xxxx: arbiter_signal = 6'b010000;
        6'b1xxxxx: arbiter_signal = 6'b100000;   
        default: arbiter_signal = 6'b000000;
        endcase
    end
endmodule

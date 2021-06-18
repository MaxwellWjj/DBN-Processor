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

module arbiter_basic_vh (
    input [`NUM_CORE_V - 1 : 0] signal_in,
    output logic [`NUM_CORE_V - 1 : 0] arbiter_signal
);

    // The comb module should be modified if the number of cores changes
    always_comb begin
        casex(signal_in)
        10'b0000000000: arbiter_signal = 10'b0000000000;
        10'b0000000001: arbiter_signal = 10'b0000000001;
        10'b000000001x: arbiter_signal = 10'b0000000010;
        10'b00000001xx: arbiter_signal = 10'b0000000100;
        10'b0000001xxx: arbiter_signal = 10'b0000001000;
        10'b000001xxxx: arbiter_signal = 10'b0000010000;
        10'b00001xxxxx: arbiter_signal = 10'b0000100000;
        10'b0001xxxxxx: arbiter_signal = 10'b0001000000;
        10'b001xxxxxxx: arbiter_signal = 10'b0010000000;
        10'b01xxxxxxxx: arbiter_signal = 10'b0100000000;     
        10'b1xxxxxxxxx: arbiter_signal = 10'b1000000000;     
        default: arbiter_signal = 10'b0000000000;
        endcase
    end
endmodule

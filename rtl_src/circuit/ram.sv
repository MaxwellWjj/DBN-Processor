//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Address generator of Transposed Memory
// Module Name: addr_gen
// Project Name: DBN Processor
// Target Devices: Zynq-7100
// Tool Versions: Vivado 2019.1
// Description:
//
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// 
//////////////////////////////////////////////////////////////////////////////////


module SdpRamRf #(
    parameter DW = 8, AW = 8, WORDS = 256
)(
    input wire clk,
    input wire [AW - 1 : 0] addr_a,
    input wire wr_a,
    input wire [DW - 1 : 0] din_a,
    input wire [AW - 1 : 0] addr_b,
    output logic [DW - 1 : 0] qout_b
);
    logic [DW - 1 : 0] ram [WORDS];
    always_ff@(posedge clk) begin
        if(wr_a) begin
            ram[addr_a] <= din_a;
        end
    end
    always_ff@(posedge clk) begin
        qout_b <= ram[addr_b];
    end
endmodule

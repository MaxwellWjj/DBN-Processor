//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: myfifo
// Module Name: addr_gen
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


module myfifo #(
	parameter DW = 8, AW = 10, WORDS = 1024
)(
	input wire clk,
	input wire rst,
	input wire [DW - 1 : 0] din,
	input wire write,
	output logic [DW - 1 : 0] dout,
	input wire read,
	output logic full, empty
);
	logic [AW - 1 : 0] wr_cnt;
	logic [AW - 1 : 0] rd_cnt;

	always_ff@(posedge clk) begin
		if(rst) wr_cnt <= 0;
		else if(write) wr_cnt <= wr_cnt + 1'b1;
	end

	always_ff@(posedge clk) begin
		if(rst) rd_cnt <= 0;
		else if(read) rd_cnt <= rd_cnt + 1'b1;
	end

	logic empty_dly;
	logic [AW - 1 : 0] data_cnt;
	assign data_cnt = wr_cnt - rd_cnt;
	assign full = (data_cnt == (WORDS - 1));
	assign empty = (data_cnt == 0);
	always_ff @( posedge clk ) begin
		empty_dly <= empty;
	end
	logic [DW - 1 : 0] d_out;
	assign dout = (~empty_dly)?d_out:0;
	SdpRamRf #(.DW(DW), .AW(AW), .WORDS(WORDS)) theRam(
        .clk(clk), .addr_a(wr_cnt), .wr_a(write),
        .din_a(din), .addr_b(rd_cnt), .qout_b(d_out)
	);

endmodule
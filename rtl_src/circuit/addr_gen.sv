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
// Description: 1- All the parameters are defined in system_define.sv file, instead of using #(parameter ...).
//              2- This module needs faster clock to generate addresses.
//
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// 
//////////////////////////////////////////////////////////////////////////////////

`include "system_define.sv"

module addr_gen (
    // Clock, reset and enable
    input wire clk_f,
	input wire rst,	
	input wire en,

    // Control
	input wire [2:0] state_signal,    

    // Input neuron states
    input wire [`NUM_VN_ONECORE - 1 : 0] v_states,
    input wire [`NUM_HN_ONECORE - 1 : 0] h_states,

    // Address generation signal
    output wire gen_vh_finished,
    output wire gen_hv_finished,

    // Fetch an address from FIFO to TWM
    input wire en_fetch, // Enable signal to fetch an address (single cycle)
    output wire [`BW_ADDR - 1 : 0] addr_fetch,
    output wire empty_fifo

);

    logic [`NUM_N_MAX_ONECORE - 1 : 0] n_states;
    always_comb begin
        if (state_signal == `ADDR_VH) begin
            n_states = v_states;
        end
        else if (state_signal == `ADDR_HV) begin
            n_states = h_states;
        end
        else n_states = 0;
    end

    // Generation phase
    logic [4 : 0] cnt_neurons; // Note that the width should be modified if the number of neurons changes.
    always_ff @( posedge clk_f ) begin
        if (rst) begin
            cnt_neurons <= 0;
        end
        else if (en) begin
            if (state_signal == `ADDR_VH) begin
                if (cnt_neurons == `NUM_VN_ONECORE) begin
                    cnt_neurons <= 0;
                end
                else begin
                    cnt_neurons <= cnt_neurons + 1;
                end
            end
            else if (state_signal == `ADDR_HV) begin
                if (cnt_neurons == `NUM_HN_ONECORE) begin
                    cnt_neurons <= 0;
                end
                else begin
                    cnt_neurons <= cnt_neurons + 1;
                end
            end
            else begin
                cnt_neurons <= 0;
            end
        end
    end

    assign gen_vh_finished = (cnt_neurons == `NUM_VN_ONECORE) & (state_signal == `ADDR_VH);
    assign gen_hv_finished = (cnt_neurons == `NUM_HN_ONECORE) & (state_signal == `ADDR_HV);

    // FIFO
    logic fifo_write;
    logic fifo_full;
    assign fifo_write = ((n_states[cnt_neurons] == 1) && (cnt_neurons < `NUM_N_MAX_ONECORE));
    myfifo #(.DW(`BW_ADDR), .AW(`AW_ADDR_FIFO), .WORDS(`DEPTH_ADDR_FIFO)) addr_buffer (
        clk_f, rst, cnt_neurons[`BW_ADDR - 1 : 0], fifo_write, addr_fetch, en_fetch, fifo_full, empty_fifo
    );

endmodule
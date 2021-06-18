//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Transposed memory of weights (multi read & single write)
// Module Name: t_mem
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

module t_mem (
    // Clock, reset and enable
    input wire clk,
	input wire rst,	
	input wire en,
    input wire init_w_new,
    // Control
	input wire [2:0] state_signal,
    input wire update_write_en,

    // Address of both read and write
    input wire [`BW_ADDR - 1 : 0] v_addr,
    input wire [`BW_ADDR - 1 : 0] h_addr_write,
    input wire [`BW_ADDR - 1 : 0] h_addr_read,

    // Input weights, including initialization and updating
	input wire signed [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] input_weights,

    // Output weights 
    output logic signed [`BW_WEIGHTS * `NUM_TM_H - 1 : 0] output_weights_vh,
    output logic signed [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] output_weights_hv

);

    logic signed [`BW_WEIGHTS - 1 : 0] mem_regs [`NUM_TM_H - 1 : 0] [`NUM_TM_V - 1 : 0];

    // Write in
    integer a;
    integer b;
    integer c;
    always_ff @( posedge clk) begin
        if (rst) begin
            for (a = 0; a < `NUM_TM_H; a = a + 1) begin
                for (b = 0; b < `NUM_TM_V; b = b + 1) begin
                    mem_regs[a][b] <= 0;
                end
            end
        end
        else if (en) begin
            if (((state_signal == `INIT)&&(init_w_new)) || ((state_signal == `UPDATE) && (update_write_en == 1'b1))) begin
                // These statements are naive version.
                // TODO: modifying them into parameters-based statements.
                mem_regs[h_addr_write][0] <= input_weights[10:0];
                mem_regs[h_addr_write][1] <= input_weights[21:11];
                mem_regs[h_addr_write][2] <= input_weights[32:22];
                mem_regs[h_addr_write][3] <= input_weights[43:33];
                mem_regs[h_addr_write][4] <= input_weights[54:44];
                mem_regs[h_addr_write][5] <= input_weights[65:55];
                mem_regs[h_addr_write][6] <= input_weights[76:66];
                mem_regs[h_addr_write][7] <= input_weights[87:77];
                mem_regs[h_addr_write][8] <= input_weights[98:88];
                mem_regs[h_addr_write][9] <= input_weights[109:99];
            end
        end
    end
        
    // Read out
    genvar d;
    generate
        for (d = 0; d < `NUM_TM_V; d = d + 1) begin
            assign output_weights_hv[(`BW_WEIGHTS * (d+1) - 1) : `BW_WEIGHTS * d] = ((state_signal == `REC_HV)||(state_signal == `UPDATE))?mem_regs[h_addr_read][d]:0;
        end    
    endgenerate

    genvar e;
    generate
        for (e = 0; e < `NUM_TM_H; e = e + 1) begin
            assign output_weights_vh[(`BW_WEIGHTS * (e+1) - 1) : `BW_WEIGHTS * e] = (state_signal == `GEN_VH)?mem_regs[e][v_addr]:0;
        end        
    endgenerate

endmodule
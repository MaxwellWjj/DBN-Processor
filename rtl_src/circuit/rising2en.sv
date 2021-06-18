//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2020/05/18
// Design Name: A Reconfigurable GALS Multi-core Design of Restricted Boltzmann Machine Based on Biologically Plausible Learning Rule
// Module Name: Rising2En
// Project Name: GALS_RBM
// Target Devices: Zynq-7100
// Tool Versions: Vivado 2018.3
// Description: 1- All signals are under 100MHz clock domain synchronously.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
//
// Additional Comments: Parameters- DW_W: weights width; DW_PARTSUM: partial sum width, DW_TOTALSUM: total sum width
//                      V_NUM: The number of visible neurons in one core, H_NUM: The number of hidden neurons in one core, 
//                      CORE_NUM: The number of cores in system, DW_DELTA: Data width of delta, DW_LFSR: DW of LFSR,
//                      TH: Threshold for sparsity, DW_T_MEM_ADDR: Address width of local memory,
//                      CORE_NUM_R: Row number of cores, CORE_NUM_C: Colunm number of cores
// 
//////////////////////////////////////////////////////////////////////////////////


module Rising2En #( parameter SYNC_STG = 1 )(
    input wire clk, in,
    output logic en, out
);
    logic [SYNC_STG : 0] dly;
    always_ff@(posedge clk) begin
        dly <= {dly[SYNC_STG - 1 : 0], in};    
    end
    assign en = (SYNC_STG ? dly[SYNC_STG -: 2] : {dly, in}) == 2'b01;
    assign out = dly[SYNC_STG];
endmodule

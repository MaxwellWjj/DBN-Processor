//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Update unit of weights with CD algorithm
// Module Name: Update_Unit_CD
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


module Update_Unit_CD (
    // Clock, reset and enable. If no pipeline in the update unit, these signals are neglected.
    input wire clk,
    input wire rst,
    input en,

    // Update signals
    input wire signed [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] input_weight,
    input wire [`NUM_VN_ONECORE - 1 : 0] v_states_0,
    input wire h_states_0,
    input wire [`NUM_VN_ONECORE - 1 : 0] v_states_2,
    input wire h_states_2,
    output logic signed [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] write_back_weight

);

    // NO PIPELINE MODE
    // TODO: Test timing for this module. If the timing violates the constraints, adding pipeline in this module! 
    logic [`NUM_VN_ONECORE - 1 : 0] vh_0;
    logic [`NUM_VN_ONECORE - 1 : 0] vh_2;
    logic [1 : 0] delta_vh [`NUM_VN_ONECORE - 1 : 0];   // 2'b00: delta = 0; 2'b01: delta = -1; 2'b10: delta = 1
    logic [`BW_WEIGHTS - 1 : 0] delta_vh_learningrate [`NUM_VN_ONECORE - 1 : 0];
    logic signed [`BW_WEIGHTS - 1 : 0] delta_vh_learningrate_signed [`NUM_VN_ONECORE - 1 : 0];

    assign vh_0 = (h_states_0) ? v_states_0 : 0;
    assign vh_2 = (h_states_2) ? v_states_2 : 0;
    
    genvar a;
    generate
        for (a = 0; a < `NUM_VN_ONECORE; a = a + 1) begin
            assign delta_vh[a] = (vh_0[a] == vh_2[a]) ? 2'b00 : (vh_0[a] == 1'b0) ? 2'b01 : 2'b10;
            assign delta_vh_learningrate[a] = (delta_vh[a] == 2'b00) ? 0 : (9'b100000000 >> `LEARNING_RATE); // TODO: modifying them into parameters-based statements.
            assign delta_vh_learningrate_signed[a] = (delta_vh[a] == 2'b00) ? 0 : (delta_vh[a] == 2'b10) ? delta_vh_learningrate[a] : (- delta_vh_learningrate[a]);
            always_ff @( posedge clk ) begin
                if (rst) begin
                    write_back_weight[(`BW_WEIGHTS * (a + 1) - 1) : `BW_WEIGHTS * a] <= 0;
                end
                else if (en) write_back_weight[(`BW_WEIGHTS * (a + 1) - 1) : `BW_WEIGHTS * a] <= input_weight[(`BW_WEIGHTS * (a + 1) - 1) : `BW_WEIGHTS * a] + delta_vh_learningrate_signed[a];
            end
        end
    endgenerate

endmodule
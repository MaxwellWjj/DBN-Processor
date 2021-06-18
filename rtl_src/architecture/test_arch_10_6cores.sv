//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: Test architecture which includes 100 VN and 30 HN
// Module Name: Test_arch_100_30
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

module test_arch_100_30 (
    // Clock, reset and enable
    input wire clk,
	input wire rst,
	input wire en,

    // Initialization
    input wire [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] init_weight,
    input wire [`BW_CORE_INDEX - 1 : 0] init_weight_index,
    input wire init_w_en,
    input wire init_w_new,
    input wire training_or_inference,
    // Input VN states
    input wire [`NUM_VN_ONECORE * `NUM_CORE_V - 1 : 0] init_v,

    // Operation
    input wire begin_operation,  // Single cycle signal, enabling the operation of RBM

    // H states of inference phase and debug signals
    input wire [`BW_CORE_INDEX - 1 : 0] out_index,
    output wire [`NUM_HN_ONECORE - 1 : 0] infer_h,
    output wire [`NUM_HN_ONECORE - 1 : 0] debug_h0_o,
    output wire [`NUM_VN_ONECORE - 1 : 0] debug_v2_o,
    output wire [`NUM_HN_ONECORE - 1 : 0] debug_h2_o

);

    // Output dones and input receives
    wire [`NUM_CORE - 1 : 0] done_vh;
    wire [`NUM_CORE - 1 : 0] done_hv;
    wire [`NUM_CORE - 1 : 0] received_vh;
    wire [`NUM_CORE - 1 : 0] received_hv;

    // Bus interface for rbm write channel vh
    wire [`BW_PS * `NUM_HN_ONECORE - 1 : 0] rbm_write_channel_data_vh [`NUM_CORE - 1 : 0];

    // Bus interface for rbm write channel hv
    wire [`BW_PS * `NUM_VN_ONECORE - 1 : 0] rbm_write_channel_data_hv [`NUM_CORE - 1 : 0];

    // New states (these signals are connected to RBM cores)
    wire [`NUM_VN_ONECORE - 1 : 0] new_states_v [`NUM_CORE- 1 : 0];
    wire [`NUM_VN_ONECORE - 1 : 0] new_states_v_en [`NUM_CORE - 1 : 0];
    wire [`NUM_HN_ONECORE - 1 : 0] new_states_h [`NUM_CORE - 1 : 0];
    wire [`NUM_HN_ONECORE - 1 : 0] new_states_h_en [`NUM_CORE - 1 : 0];

    // New states (these signals are connected to AGS cores)
    wire [`NUM_VN_ONECORE - 1 : 0] ags_new_states_v [`NUM_CORE_V - 1 : 0];
    wire [`NUM_VN_ONECORE - 1 : 0] ags_new_states_v_en [`NUM_CORE_V - 1 : 0];
    wire [`NUM_HN_ONECORE - 1 : 0] ags_new_states_h [`NUM_CORE_H - 1 : 0];
    wire [`NUM_HN_ONECORE - 1 : 0] ags_new_states_h_en [`NUM_CORE_H - 1 : 0];

    // Debugging signals, including h_0, v_2, h_2 in the training mode
    wire [`NUM_HN_ONECORE - 1 : 0] debug_h0 [`NUM_CORE - 1 : 0];
    wire [`NUM_VN_ONECORE - 1 : 0] debug_v2 [`NUM_CORE - 1 : 0];
    wire [`NUM_HN_ONECORE - 1 : 0] debug_h2 [`NUM_CORE - 1 : 0];

    // Signals for arbiter
    wire [`NUM_CORE_H - 1 : 0] data_in_en_vh;
    wire [`BW_PS * `NUM_HN_ONECORE - 1 : 0] data_in_vh [`NUM_CORE_H - 1 : 0];
    wire [`NUM_CORE_V - 1 : 0] data_in_en_hv;
    wire [`BW_PS * `NUM_VN_ONECORE - 1 : 0] data_in_hv [`NUM_CORE_V - 1 : 0];

    // Final weights 
    wire [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] final_weight [`NUM_CORE - 1 : 0];

    // RBM cores
    wire [`NUM_CORE - 1 : 0] init_w_en_core;
    wire [`NUM_HN_ONECORE - 1 : 0] infer_h_one_core [`NUM_CORE - 1 : 0];
    wire [`NUM_VN_ONECORE - 1 : 0] init_v_one_core [`NUM_CORE - 1 : 0];

    genvar a;
    generate
        for (a = 0; a < `NUM_CORE; a = a + 1) begin
            rbm_core rbmcores (clk, rst, en, init_weight, training_or_inference, 1'b0, begin_operation, init_v_one_core[a], init_w_en_core[a], init_w_new, infer_h_one_core[a], 
                                final_weight[a], done_vh[a], done_hv[a], received_vh[a], received_hv[a], rbm_write_channel_data_vh[a],
                                rbm_write_channel_data_hv[a], new_states_v[a], new_states_v_en[a], new_states_h[a], new_states_h_en[a],
                                debug_h0[a], debug_v2[a], debug_h2[a]);
            assign init_w_en_core[a] = (init_weight_index == a) ? init_w_en : 1'b0;
        end
    endgenerate

    assign infer_h = infer_h_one_core[out_index];
    assign debug_h0_o = debug_h0[out_index];
    assign debug_v2_o = debug_v2[out_index];
    assign debug_h2_o = debug_h2[out_index];

    genvar j;
    genvar k;
    generate
        for (j = 0; j < `NUM_CORE_V; j = j + 1) begin
            for (k = 0; k < `NUM_CORE_H; k = k + 1) begin
                assign init_v_one_core[j + `NUM_CORE_V * k] = init_v[`NUM_VN_ONECORE * (j + 1) - 1 : `NUM_VN_ONECORE * j];                
            end
        end
    endgenerate

    // Arbiter and AGS cores in v->h phase
    wire [`NUM_CORE_V - 1 : 0] done_vh_temp [`NUM_CORE_H - 1 : 0];
    wire [`BW_PS * `NUM_HN_ONECORE * `NUM_CORE_V - 1 : 0] rbm_write_channel_data_vh_temp [`NUM_CORE_H - 1 : 0];
    wire [`NUM_CORE_V - 1 : 0] received_vh_temp [`NUM_CORE_H - 1 : 0];

    genvar b;
    genvar l;
    generate
        for (b = 0; b < `NUM_CORE_H; b = b + 1) begin
            // TODO: modifying this statement as paramiterized one.
            arbiter_vh arb_vh (clk, rst, en, done_vh_temp[b], rbm_write_channel_data_vh_temp[b], data_in_en_vh[b], 
                                received_vh_temp[b], data_in_vh[b]);
            for (l = 0; l < `NUM_CORE_V; l = l + 1) begin
                assign done_vh_temp[b][l] = done_vh[b * `NUM_CORE_V + l];
                assign rbm_write_channel_data_vh_temp[b][(`BW_PS * `NUM_HN_ONECORE * (l + 1) - 1) : `BW_PS * `NUM_HN_ONECORE * l] = rbm_write_channel_data_vh[b * `NUM_CORE_V + l];
                assign received_vh[b * `NUM_CORE_V + l] = received_vh_temp[b][l];
            end
        end
    endgenerate

    genvar c;
    genvar d;
    genvar e;
    generate
        for (c = 0; c < `NUM_CORE_H; c = c + 1) begin
            for (d = 0; d < `NUM_HN_ONECORE; d = d + 1) begin
                ags_core_vh ags_vh (clk, rst, en, data_in_en_vh[c], data_in_vh[c][(`BW_PS * (d + 1) - 1) : `BW_PS * d], ags_new_states_h[c][d],
                                    ags_new_states_h_en[c][d]);
                for (e = 0; e < `NUM_CORE_V; e = e + 1) begin
                    assign new_states_h[`NUM_CORE_V * c + e][d] = ags_new_states_h[c][d];
                    assign new_states_h_en[`NUM_CORE_V * c + e][d] = ags_new_states_h_en[c][d];
                end
            end            
        end
    endgenerate

    // Arbiter and AGS cores in h->v phase
    wire [`NUM_CORE_H - 1 : 0] done_hv_temp [`NUM_CORE_V - 1 : 0];
    wire [`BW_PS * `NUM_VN_ONECORE * `NUM_CORE_H - 1 : 0] rbm_write_channel_data_hv_temp [`NUM_CORE_V - 1 : 0];
    wire [`NUM_CORE_H - 1 : 0] received_hv_temp [`NUM_CORE_V - 1 : 0];

    genvar f;
    genvar m;
    generate
        for (f = 0; f < `NUM_CORE_V; f = f + 1) begin
            // TODO: need to be modified carefully!
            arbiter_hv arb_hv (clk, rst, en, done_hv_temp[f], rbm_write_channel_data_hv_temp[f], data_in_en_hv[f], 
                                received_hv_temp[f], data_in_hv[f]);
            for (m = 0; m < `NUM_CORE_H; m = m + 1) begin
                assign done_hv_temp[f][m] = done_hv[m * `NUM_CORE_V + f];
                assign rbm_write_channel_data_hv_temp[f][(`BW_PS * `NUM_VN_ONECORE * (m + 1) - 1) : `BW_PS * `NUM_VN_ONECORE * m] = rbm_write_channel_data_hv[m * `NUM_CORE_V + f];
                assign received_hv[m * `NUM_CORE_V + f] = received_hv_temp[f][m];                
            end
        end
    endgenerate

    genvar g;
    genvar h;
    genvar i;
    generate
        for (g = 0; g < `NUM_CORE_V; g = g + 1) begin
            for (h = 0; h < `NUM_VN_ONECORE; h = h + 1) begin
                ags_core_hv ags_hv (clk, rst, en, data_in_en_hv[g], data_in_hv[g][(`BW_PS * (h + 1) - 1) : `BW_PS * h], ags_new_states_v[g][h],
                                    ags_new_states_v_en[g][h]);
                for (i = 0; i < `NUM_CORE_H; i = i + 1) begin
                    assign new_states_v[`NUM_CORE_V * i + g][h] = ags_new_states_v[g][h];
                    assign new_states_v_en[`NUM_CORE_V * i + g][h] = ags_new_states_v_en[g][h];
                end
            end
        end
    endgenerate

endmodule
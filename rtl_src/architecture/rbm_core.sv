//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: RBM Core
// Module Name: rbm_core
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

module rbm_core (
    // Clock, reset and enable
    input wire clk,
	input wire rst,
	input wire en,

    // Initialization
    input wire [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] init_weight,

    // Operation    
    input wire training_or_inference,    // 0- training; 1- inference
    input wire CD_or_VPF, // 0- CD; 1- VPF
    input wire begin_operation,  // Single cycle signal, enabling the operation of RBM

    // Input neuron states
    input wire [`NUM_VN_ONECORE - 1 : 0] init_v,
    input wire init_w_en,
    input wire init_w_new,

    // Output neuron states for inference (to the next RBM as input)
    output logic [`NUM_HN_ONECORE - 1 : 0] infer_h,

    // Output weights
    output wire [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] final_weight,

    // Output dones and input receives
    output logic done_vh,
    output logic done_hv,
    input wire received_vh,
    input wire received_hv,

    // Bus interface for rbm write channel vh
    output wire signed [`BW_PS * `NUM_HN_ONECORE - 1 : 0] rbm_write_channel_data_vh,

    // Bus interface for rbm write channel hv
    output wire signed [`BW_PS * `NUM_VN_ONECORE - 1 : 0] rbm_write_channel_data_hv,

    // New states from AGS cores
    input wire [`NUM_VN_ONECORE - 1 : 0] new_states_v,
    input wire [`NUM_VN_ONECORE - 1 : 0] new_states_v_en,
    input wire [`NUM_HN_ONECORE - 1 : 0] new_states_h,
    input wire [`NUM_HN_ONECORE - 1 : 0] new_states_h_en,

    // Debugging signals, including h_0, v_2, h_2 in the training mode
    output wire [`NUM_HN_ONECORE - 1 : 0] debug_h0,
    output wire [`NUM_VN_ONECORE - 1 : 0] debug_v2,
    output wire [`NUM_HN_ONECORE - 1 : 0] debug_h2
    
);
    
    // FSM
    logic [2:0] fsm_state;
    logic [2:0] fsm_next_state;
    logic [2:0] fsm_state_dly;
    always_ff @(posedge clk) begin
        if(rst) begin
            fsm_state <= `IDLE;
        end
        else if(en) fsm_state <= fsm_next_state;        
    end

    logic init_finished;
    logic vh_addr_finished;
    logic hv_addr_finished;
    logic vh_finished;
    logic hv_finished;
    logic update_finished;
    logic [2 : 0] cnt_vh; // This counter is used for counting the times of gen_vh operation
    always_comb begin
        case(fsm_state)
            `IDLE: begin
                if(en) begin
                    if (init_w_en&(~init_finished)) begin
                        fsm_next_state = `INIT;
                    end
                    else if (begin_operation) begin
                        fsm_next_state = `ADDR_VH;
                    end
                    else fsm_next_state = fsm_state;
                end
                else fsm_next_state = fsm_state;
            end
            `INIT: begin
                if(en) begin
                    if(init_finished) fsm_next_state = `IDLE;
                    else fsm_next_state = fsm_state;
                end
                else fsm_next_state = fsm_state;
            end
            `ADDR_VH: begin
                if (en) begin
                    if (vh_addr_finished) begin
                        fsm_next_state = `GEN_VH;
                    end
                    else fsm_next_state = fsm_state;
                end
                else fsm_next_state = fsm_state;
            end
            `GEN_VH: begin
                if(en) begin
                    if(vh_finished) begin
                        if (training_or_inference) begin
                            fsm_next_state = `IDLE;
                        end
                        else begin
                            if ((cnt_vh == 3'd2) && (~CD_or_VPF)) begin
                                fsm_next_state = `UPDATE;
                            end
                            else begin 
                                fsm_next_state = `ADDR_HV;
                            end
                        end
                    end
                    else begin 
                        fsm_next_state = fsm_state;
                    end
                end
                else begin 
                    fsm_next_state = fsm_state;
                end
            end
            `ADDR_HV: begin
                if (en) begin
                    if (hv_addr_finished) begin
                        fsm_next_state = `REC_HV;
                    end
                    else fsm_next_state = fsm_state;
                end
                else fsm_next_state = fsm_state;
            end
            `REC_HV: begin
                if(en) begin
                    if(hv_finished) begin
                        if(CD_or_VPF) begin
                            fsm_next_state = `UPDATE;
                        end
                        else begin 
                            fsm_next_state = `ADDR_VH;
                        end
                    end
                    else begin 
                        fsm_next_state = fsm_state;
                    end
                end
                else begin
                    fsm_next_state = fsm_state;
                end
            end
            `UPDATE: begin
                if(en) begin
                    if(update_finished) begin
                        fsm_next_state = `IDLE;
                    end
                    else fsm_next_state = fsm_state;
                end
                else begin
                    fsm_next_state = fsm_state;
                end
            end            
            default: begin
                fsm_next_state = `IDLE;
            end
        endcase        
    end

    always_ff @( posedge clk ) begin
        if (rst) begin
            cnt_vh <= 0;
        end
        else if (en) begin
            if (vh_finished) begin
                cnt_vh <= cnt_vh + 1;
            end
            else if (fsm_next_state == `UPDATE) begin
                cnt_vh <= 0;
            end
        end
    end

    always_ff @( posedge clk ) begin
        if (rst) begin
            fsm_state_dly <= 0;
        end
        else if (en) begin
            fsm_state_dly <= fsm_state;
        end
    end

    // States register
    logic [`NUM_VN_ONECORE - 1 : 0] v_states;
    logic [`NUM_HN_ONECORE - 1 : 0] h_states;

    // States buffer (for CD algorithm)
    logic [`NUM_VN_ONECORE - 1 : 0] v_states_0;
    logic [`NUM_HN_ONECORE - 1 : 0] h_states_0;
    
    // Partial and total summation
    logic signed [`BW_PS - 1 : 0] ps_vh [`NUM_HN_ONECORE - 1 : 0];
    logic signed [`BW_PS - 1 : 0] ps_hv [`NUM_VN_ONECORE - 1 : 0];

    // Pre-generating addresses and initialization addresses
    logic en_fetch;
    logic empty_fifo;
    logic empty_fifo_dly;
    logic [`BW_ADDR - 1 : 0] haddr_init;
    logic [`BW_ADDR - 1 : 0] init_count;
    logic [`BW_ADDR - 1 : 0] addr_fetch;  
    addr_gen addrgen (clk, rst, en, fsm_state, v_states, h_states, vh_addr_finished, hv_addr_finished, en_fetch, addr_fetch, empty_fifo);

    always_ff @( posedge clk ) begin
        if (rst) begin
            haddr_init <= 0;
            init_count <= 0;
        end
        else if (en) begin
            if ((fsm_state == `INIT) && init_w_new) begin
                init_count <= init_count + 1;
                if(haddr_init < `NUM_HN_ONECORE - 1)
                    haddr_init <= haddr_init + 1;
            end
        end
    end
    assign init_finished = (init_count == `NUM_HN_ONECORE);

    assign en_fetch = ((fsm_state == `GEN_VH || fsm_state == `REC_HV) && (empty_fifo == 0));

    // Transposed memory and PEs (accumulators)
    logic [`BW_ADDR - 1 : 0] v_addr;
    logic [`BW_ADDR - 1 : 0] h_addr; // Read
    logic [`BW_ADDR - 1 : 0] h_addr_write; // Write
    logic [`BW_ADDR - 1 : 0] h_addr_update_write; // Update write
    logic [`BW_ADDR - 1 : 0] h_addr_update_read; // Update read
    logic update_write_en; // In update phase, enabling write.
    logic [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] write_back_weight;
    logic [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] input_weight_twm;
    logic signed [`BW_WEIGHTS * `NUM_TM_H - 1 : 0] output_weights_vh;
    logic signed [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] output_weights_hv;
    logic signed [`BW_WEIGHTS * `NUM_TM_H - 1 : 0] output_weights_vh_pe;
    logic signed [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] output_weights_hv_pe;
    // The signals here are mixed with "update unit"
    assign h_addr = (fsm_state == `REC_HV) ? addr_fetch : (fsm_state == `UPDATE) ? h_addr_update_read : 0;
    assign h_addr_write = (fsm_state == `INIT) ? haddr_init : (fsm_state_dly == `UPDATE) ? h_addr_update_write : 0;
    assign v_addr = (fsm_state == `GEN_VH) ? addr_fetch : 0;
    assign input_weight_twm = (fsm_state == `INIT) ? init_weight : (fsm_state_dly == `UPDATE) ? write_back_weight : 0;
    assign update_write_en = (fsm_state_dly == `UPDATE);
    t_mem trans_w_mem (clk, rst, en, init_w_new, fsm_state, update_write_en, v_addr, h_addr_write, h_addr, input_weight_twm, output_weights_vh, output_weights_hv);

    always_ff @( posedge clk ) begin
        if (rst) begin
            empty_fifo_dly <= 0;
        end
        else if (en) begin
            empty_fifo_dly <= empty_fifo;
        end
    end

    assign output_weights_vh_pe = (fsm_state_dly == `GEN_VH) ? output_weights_vh : 0;
    assign output_weights_hv_pe = (fsm_state_dly == `REC_HV) ? output_weights_hv : 0;

    genvar a;
    generate
        for (a = 0; a < `NUM_HN_ONECORE; a = a + 1) begin
            PE acc_vh (clk, rst, (en & (empty_fifo_dly == 0) & (fsm_state == `GEN_VH)), fsm_state, output_weights_vh_pe[(`BW_WEIGHTS * (a + 1) - 1) : `BW_WEIGHTS * a], ps_vh[a]);
        end
    endgenerate

    genvar b;
    generate
        for (b = 0; b < `NUM_VN_ONECORE; b = b + 1) begin
            PE acc_hv (clk, rst, (en & (empty_fifo_dly == 0) & (fsm_state == `REC_HV)), fsm_state, output_weights_hv_pe[(`BW_WEIGHTS * (b + 1) - 1) : `BW_WEIGHTS * b], ps_hv[b]);
        end
    endgenerate

    // Done controller
    logic flag_done_vh;
    logic flag_done_hv;    
//    logic flag_delay;
//    logic [2:0] delay_conut;
//    always_ff @( posedge clk ) begin
//        if( rst )begin
//            delay_conut <= 0;
//        end
//        else begin
//            if((fsm_state == `GEN_VH) || (fsm_state == `REC_HV)) begin
//                if((delay_conut < 3'd3)) begin
//                    delay_conut <= delay_conut + 3'd1;
//                end
//            end
//            else begin
//                delay_conut <= 0;
//            end
//        end
//    end
//    assign flag_delay = (delay_conut == 3'd3);
    always_ff @( posedge clk ) begin
        if (rst | begin_operation) begin
            flag_done_vh <= 1'b0;
        end
        else if (en) begin
            if (received_vh) begin
                flag_done_vh <= 1'b1;
            end
            else if (vh_finished) begin
                flag_done_vh <= 1'b0;
            end
        end
    end

    always_ff @( posedge clk ) begin
        if (rst) begin
            done_vh <= 1'b0;
        end
        else if (en) begin
            if ((fsm_state == `GEN_VH) && (empty_fifo == 1'b1)) begin
                if ((~received_vh)&&(~flag_done_vh)) begin
                    done_vh <= 1'b1;
                end
                else if (received_vh) begin
                    done_vh <= 1'b0;
                end
                else done_vh <= 1'b0;
            end
            else done_vh <= 1'b0;
        end
    end

    always_ff @( posedge clk ) begin
        if (rst | begin_operation) begin
            flag_done_hv <= 1'b0;
        end
        else if (en) begin
            if (received_hv) begin
                flag_done_hv <= 1'b1;
            end
            else if (hv_finished) begin
                flag_done_hv <= 1'b0;
            end
        end
    end

    always_ff @( posedge clk ) begin
        if (rst) begin
            done_hv <= 1'b0;
        end
        else if (en) begin
            if ((fsm_state == `REC_HV) && (empty_fifo == 1'b1)) begin
                if ((~received_hv)&&(~flag_done_hv)) begin
                    done_hv <= 1'b1;
                end
                else if (received_hv) begin
                    done_hv <= 1'b0;
                end
                else done_hv <= 1'b0;
            end
            else done_hv <= 1'b0;
        end
    end

    // GEN_VH phase, RBM core -> AGS core
    genvar c;
    generate
        for (c = 0; c < `NUM_HN_ONECORE; c = c + 1) begin
            assign rbm_write_channel_data_vh[(`BW_PS * (c + 1) - 1) : `BW_PS * c] = (done_vh == 1'b1) ? ps_vh[c] : 0;
        end
    endgenerate

    // REC_HV phase, RBM core -> AGS core
    genvar d;
    generate
        for (d = 0; d < `NUM_VN_ONECORE; d = d + 1) begin
            assign rbm_write_channel_data_hv[(`BW_PS * (d + 1) - 1) : `BW_PS * d] = (done_hv == 1'b1) ? ps_hv[d] : 0;
        end
    endgenerate


    // Signal of GEN_VH and REC_HV phases finishing
    logic [`NUM_HN_ONECORE - 1 : 0] newstate_vh_finished;
    genvar g;
    generate
        for (g = 0; g < `NUM_VN_ONECORE; g = g + 1) begin
            always_ff @( posedge clk ) begin
                if (rst) begin
                    newstate_vh_finished[g] <= 0;
                end
                else if (en) begin
                    if (fsm_state == `GEN_VH) begin
                        if (new_states_h_en[g] == 1'b1) begin
                            newstate_vh_finished[g] <= 1'b1;
                        end
                    end
                    else newstate_vh_finished[g] <= 0;
                end
            end
        end
    endgenerate
    assign vh_finished = (newstate_vh_finished == (2**`NUM_HN_ONECORE) - 1);

    logic [`NUM_VN_ONECORE - 1 : 0] newstate_hv_finished;
    genvar h;
    generate
        for (h = 0; h < `NUM_VN_ONECORE; h = h + 1) begin
            always_ff @( posedge clk ) begin
                if (rst) begin
                    newstate_hv_finished[h] <= 0;
                end
                else if (en) begin
                    if (fsm_state == `REC_HV) begin
                        if (new_states_v_en[h] == 1'b1) begin
                            newstate_hv_finished[h] <= 1'b1;
                        end
                    end
                    else newstate_hv_finished[h] <= 0;
                end
            end
        end
    endgenerate
    assign hv_finished = (newstate_hv_finished == (2**`NUM_VN_ONECORE) - 1);

    // Initialization / update of neuron states
    genvar i;
    generate
        for (i = 0; i < `NUM_VN_ONECORE; i = i + 1) begin
            always_ff @(posedge clk) begin
                if (rst) begin
                    v_states[i] <= 0;
                    v_states_0[i] <= 0;
                end
                else if(begin_operation) begin
                    v_states[i] <= init_v[i];
                    v_states_0[i] <= init_v[i];
                
                end
                else if(new_states_v_en[i]) begin
                    v_states[i] <= new_states_v[i];
                end
            end        
        end
    endgenerate

    genvar j;
    generate
        for (j = 0; j < `NUM_HN_ONECORE; j = j + 1) begin
            always_ff @(posedge clk) begin
                if (rst) begin
                    h_states[j] <= 0;
                end
                else if(new_states_h_en[j]) begin
                    h_states[j] <= new_states_h[j];
                end
            end    
        end
    endgenerate

    always_ff @( posedge clk ) begin
        if (rst) begin
            h_states_0 <= 0;
        end
        else if (en) begin
            if ((cnt_vh == 3'd1) && (vh_finished)) begin
                h_states_0 <= h_states;
            end
        end
    end

    // Update weights and control
    // TODO: If we add pipeline in update unit, we should add some logic circuits to control the h_addr_update_read and h_addr_update_write signal.
    logic [`BW_WEIGHTS * `NUM_TM_V - 1 : 0] input_weight_update;
    logic selected_h_state_0;
    logic selected_h_state_2;
    logic [4 : 0] cnt_update_haddr; // Note that the width should be modified if the number of neurons changes.

    assign input_weight_update = (fsm_state == `UPDATE) ? output_weights_hv : 0;
    always_ff @( posedge clk ) begin
        if (rst) begin
            cnt_update_haddr <= 0;
        end
        else if (fsm_state == `UPDATE) begin
            if (cnt_update_haddr == `NUM_HN_ONECORE - 1) begin
                cnt_update_haddr <= 0;
            end
            else begin
                cnt_update_haddr <= cnt_update_haddr + 1;
            end
        end
        else cnt_update_haddr <= 0;
    end
    wire update_finished_temp;
    always_ff @(posedge clk) begin
        if (rst) begin
            update_finished <= 1'b0;
        end
        else if (en) begin
            update_finished <= update_finished_temp;
        end
    end
    assign update_finished_temp = (cnt_update_haddr == `NUM_HN_ONECORE - 1);
    assign h_addr_update_read = cnt_update_haddr[`BW_ADDR - 1 : 0];

    assign selected_h_state_0 = (fsm_state == `UPDATE) ? h_states_0[h_addr_update_read] : 1'b0;
    assign selected_h_state_2 = (fsm_state == `UPDATE) ? h_states[h_addr_update_read] : 1'b0;

    always_ff @( posedge clk ) begin
        if (rst) begin
            h_addr_update_write <= 0;
        end
        else if (en) begin
            h_addr_update_write <= h_addr_update_read;
        end
    end

    Update_Unit_CD update_unit (clk, rst, en, input_weight_update, v_states_0, selected_h_state_0, v_states, selected_h_state_2, write_back_weight);

    // Output neuron states, including inference and debugging (training mode)
    always_ff @( posedge clk ) begin
        if (rst) begin
            infer_h <= 0;
        end
        else if (en) begin
            if (training_or_inference && vh_finished) begin
                infer_h <= h_states;
            end
        end
    end

    assign debug_h0 = (~training_or_inference) ? h_states_0 : 0;
    assign debug_v2 = (~training_or_inference) ? v_states : 0;
    assign debug_h2 = (~training_or_inference) ? h_states : 0;

endmodule
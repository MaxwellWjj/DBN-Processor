//////////////////////////////////////////////////////////////////////////////////
// Company: Research Lab of Ultra Low-Power and Intelligent Integrated Circuits, HUST, China
// Engineer: Jiajun Wu
// 
// Create Date: 2021/03/08
// Design Name: AGS Core
// Module Name: ags_core_vh
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

module arbiter_hv (
    // Clock, reset and enable
    input wire clk,
	input wire rst,
	input wire en,

    // From RBM core
    input wire [`NUM_CORE_H - 1 : 0] done,
    input wire [`BW_PS * `NUM_VN_ONECORE * `NUM_CORE_H - 1 : 0] partial_sum,

    // Control signals
    output wire data_in_en, // To AGS core
    output logic [`NUM_CORE_H - 1 : 0] receive, // To RBM core
    output wire [`BW_PS * `NUM_VN_ONECORE - 1 : 0] data_in // To AGS core

);

    wire [`NUM_CORE_H - 1 : 0] arbiter_temp;
    arbiter_basic_hv priority_arb_hv (done, arbiter_temp);

    // @NOTE: These statements cannot be presented as parameterized ones. We should modify them carefully if the parameters change.
//    assign data_in_en = (arbiter_temp == 10'b0000000000) ? 1'b0 : (arbiter_temp == 10'b0000000001) ? (done[0] & receive[0]): 
//                        (arbiter_temp == 10'b0000000010) ? (done[1] & receive[1]) : (arbiter_temp == 10'b0000000100) ? (done[2] & receive[2]): 
//                        (arbiter_temp == 10'b0000001000) ? (done[3] & receive[3]) : (arbiter_temp == 10'b0000010000) ? (done[4] & receive[4]):
//                        (arbiter_temp == 10'b0000100000) ? (done[5] & receive[5]) : (arbiter_temp == 10'b0001000000) ? (done[6] & receive[6]):
//                        (arbiter_temp == 10'b0010000000) ? (done[7] & receive[7]) : (arbiter_temp == 10'b0100000000) ? (done[8] & receive[8]):
//                        (arbiter_temp == 10'b1000000000) ? (done[9] & receive[9]) : 1'b0;
 assign data_in_en = (arbiter_temp == 6'b000000) ? 1'b0 : (arbiter_temp == 6'b000001) ? (done[0]): 
                        (arbiter_temp == 6'b000010) ? (done[1]) : (arbiter_temp == 6'b000100) ? (done[2]): 
                        (arbiter_temp == 6'b001000) ? (done[3]) : (arbiter_temp == 6'b010000) ? (done[4]):
                        (arbiter_temp == 6'b100000) ? (done[5]) : 1'b0;
    assign data_in = (arbiter_temp == 6'b000000) ? 0 : (arbiter_temp == 6'b000001) ? partial_sum[`BW_PS * `NUM_VN_ONECORE - 1 : 0] : 
            (arbiter_temp == 6'b000010) ? partial_sum[`BW_PS * `NUM_VN_ONECORE * 2 - 1 : `BW_PS * `NUM_VN_ONECORE] : 
            (arbiter_temp == 6'b000100) ? partial_sum[`BW_PS * `NUM_VN_ONECORE * 3 - 1 : `BW_PS * `NUM_VN_ONECORE * 2] : 
            (arbiter_temp == 6'b001000) ? partial_sum[`BW_PS * `NUM_VN_ONECORE * 4 - 1 : `BW_PS * `NUM_VN_ONECORE * 3] :
            (arbiter_temp == 6'b010000) ? partial_sum[`BW_PS * `NUM_VN_ONECORE * 5 - 1 : `BW_PS * `NUM_VN_ONECORE * 4] :
            (arbiter_temp == 6'b100000) ? partial_sum[`BW_PS * `NUM_VN_ONECORE * 6 - 1 : `BW_PS * `NUM_VN_ONECORE * 5] : 0;

    // Driving "receive" signal to inform the RBM cores that AGS core has received the data.
    // @NOTE: These statements cannot be presented as parameterized ones. We should modify them carefully if the parameters change.
//    always_ff @( posedge clk ) begin
//        if (rst) begin
//            receive <= 10'b0000000000;
//        end
//        else if (en) begin
//            if (arbiter_temp == 10'b0000000001) begin
//                receive <= 10'b0000000001;
//            end
//            else if (arbiter_temp == 10'b0000000010) begin
//                receive <= 10'b0000000010;                
//            end
//            else if (arbiter_temp == 10'b0000000100) begin
//                receive <= 10'b0000000100;                
//            end
//            else if (arbiter_temp == 10'b0000001000) begin
//                receive <= 10'b0000001000;                
//            end
//            else if (arbiter_temp == 10'b0000010000) begin
//                receive <= 10'b0000010000;                
//            end
//            else if (arbiter_temp == 10'b0000100000) begin
//                receive <= 10'b0000100000;                
//            end
//            else if (arbiter_temp == 10'b0001000000) begin
//                receive <= 10'b0001000000;                
//            end
//            else if (arbiter_temp == 10'b0010000000) begin
//                receive <= 10'b0010000000;                
//            end
//            else if (arbiter_temp == 10'b0100000000) begin
//                receive <= 10'b0100000000;                
//            end
//            else if (arbiter_temp == 10'b1000000000) begin
//                receive <= 10'b1000000000;                
//            end
//            else begin
//                receive <= 10'b0000000000;
//            end
//        end
//    end
      always_comb begin
            case(arbiter_temp)
                6'b000000 : receive = 6'b000000;
                6'b000001 : receive = 6'b000001;
                6'b000010 : receive = 6'b000010;
                6'b000100 : receive = 6'b000100;
                6'b001000 : receive = 6'b001000;
                6'b010000 : receive = 6'b010000;
                6'b100000 : receive = 6'b100000;
                default : receive = 6'b000000;
            endcase
      end

endmodule

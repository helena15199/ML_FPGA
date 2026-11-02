// -----------------------------------------------------------------------------
// axi_input.sv
// Receives an input vector over AXI-Stream as signed AXIS_DATA_W-bit words.
// The external stream protocol intentionally mirrors the UART version:
//   - one input frame = exactly INPUT_SIZE words
//   - each external word is 16 bits by default
//   - the useful payload is kept in the LSBs and resized to ACT_W bits
//   - the frame is always written into RAM bank A
//
// This module assumes ACT_W <= AXIS_DATA_W.
// IMPORTANT: s_axis_s2mm_tlast is intentionally NOT used to close the frame.
// The frame closes only after INPUT_SIZE accepted words. For the current model,
// the PC/PYNQ side sends exactly N0 = 6 words. TLAST may still be asserted by
// the DMA on the 6th word, but the RTL counter remains the source of truth.
// -----------------------------------------------------------------------------

module axi_input #(
    parameter int ACT_W       = 16,
    parameter int INPUT_SIZE  = 64,
    parameter int AW          = (INPUT_SIZE > 1) ? $clog2(INPUT_SIZE) : 1,
    parameter int AXIS_DATA_W = 16,
    parameter int AXIS_KEEP_W = (AXIS_DATA_W + 7) / 8
)(
    input  logic                       clk,
    input  logic                       rstn,

    input  logic [AXIS_DATA_W-1:0]     s_axis_s2mm_tdata,
    input  logic [AXIS_KEEP_W-1:0]     s_axis_s2mm_tkeep,
    input  logic                       s_axis_s2mm_tlast,
    input  logic                       s_axis_s2mm_tvalid,
    output logic                       s_axis_s2mm_tready,

    // High only when RAM banks A/B are available.
    // In the sequential architecture, a new frame is not accepted while the MLP
    // is computing or while the previous output frame is being sent.
    input  logic                       accept_enable,

    // Pulse from the MLP engine when the frame loaded into RAM bank A is consumed.
    input  logic                       consume_pulse,
    output logic                       frame_full,
    output logic                       loading,

    // Write interface to the vector RAM bank.
    // The input frame is always loaded into bank A, so wr_bank_sel = 0.
    output logic                       wr_bank_sel,
    output logic                       wr_en,
    output logic [AW-1:0]              wr_addr,
    output logic signed [ACT_W-1:0]    wr_data
);

    logic [AW-1:0] wr_ptr;
    logic          axis_fire;

    logic signed [AXIS_DATA_W-1:0] axis_word;

    assign axis_word = $signed(s_axis_s2mm_tdata);

    // Keep AXI sideband ports for DMA compatibility, but do not use them to
    // delimit the input frame. This avoids changing behavior when vector size
    // is generated from scales.vh.
    wire unused_axis_input_sideband = &{1'b0, s_axis_s2mm_tkeep, s_axis_s2mm_tlast};

    assign s_axis_s2mm_tready = accept_enable && !frame_full;
    assign axis_fire          = s_axis_s2mm_tvalid && s_axis_s2mm_tready;

    assign wr_bank_sel = 1'b0; // bank A
    assign wr_en       = axis_fire;
    assign wr_addr     = wr_ptr;

    // Fixed AXI input protocol: signed AXIS_DATA_W-bit words, 16 bits by default.
    // If ACT_W is smaller than AXIS_DATA_W, only the ACT_W least significant bits are kept.
    // This matches uart_input: the host must pack the useful payload in the LSBs.
    generate
        if (ACT_W == AXIS_DATA_W) begin : g_axis_rx_same_width
            assign wr_data = axis_word;
        end else begin : g_axis_rx_truncate
            assign wr_data = $signed(axis_word[ACT_W-1:0]);
        end
    endgenerate

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            frame_full <= 1'b0;
            loading    <= 1'b0;
            wr_ptr     <= '0;
        end else begin
            if (consume_pulse) begin
                frame_full <= 1'b0;
            end

            if (axis_fire) begin
                loading <= 1'b1;

                if ($unsigned(wr_ptr) == (INPUT_SIZE-1)) begin
                    frame_full <= 1'b1;
                    loading    <= 1'b0;
                    wr_ptr     <= '0;
                end else begin
                    wr_ptr <= wr_ptr + AW'(1);
                end
            end
        end
    end

    // s_axis_s2mm_tkeep and s_axis_s2mm_tlast are intentionally ignored for
    // the internal frame length. The software/DMA side must send exactly
    // INPUT_SIZE words. For the current generated MLP, INPUT_SIZE = `N0 = 6.

endmodule

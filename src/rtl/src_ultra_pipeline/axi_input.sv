// -----------------------------------------------------------------------------
// axi_input.sv
// Receives one input frame over AXI-Stream and writes it into the input
// boundary ping-pong buffer.
//
// Internal bank convention:
//   - phase_sel is the bank consumed by the current/next phase.
//   - when no phase is active, the next frame is loaded into phase_sel.
//   - while a phase is active, the next frame is loaded into ~phase_sel.
//   - consume_pulse is observed after phase_sel has toggled, so the bank that
//     has just been consumed is ~phase_sel.
//
// AXI convention:
//   - one input frame is exactly INPUT_SIZE signed ACT_W-bit words
//   - tlast is ignored for frame closing
//   - tkeep is kept only for DMA compatibility
// -----------------------------------------------------------------------------

module axi_input #(
    parameter int ACT_W      = 16,
    parameter int INPUT_SIZE = 64,
    parameter int AW         = (INPUT_SIZE > 1) ? $clog2(INPUT_SIZE) : 1,
    parameter int AXIS_DATA_W = 16,
    parameter int AXIS_KEEP_W = (AXIS_DATA_W + 7) / 8
)(
    input  logic                 clk,
    input  logic                 rstn,

    input  logic [AXIS_DATA_W-1:0]        s_axis_s2mm_tdata,
    input  logic [AXIS_KEEP_W-1:0]           s_axis_s2mm_tkeep,
    input  logic                 s_axis_s2mm_tlast,
    input  logic                 s_axis_s2mm_tvalid,
    output logic                 s_axis_s2mm_tready,

    input  logic                 phase_sel,
    input  logic                 phase_active,
    input  logic                 consume_pulse,
    output logic [1:0]           bank_full,

    output logic                 wr_bank_sel,
    output logic                 wr_en,
    output logic [AW-1:0]        wr_addr,
    output logic signed [ACT_W-1:0] wr_data
);

    logic          loading;
    logic          fill_bank_reg;
    logic [AW-1:0] wr_ptr;

    logic          target_bank;
    logic          active_fill_bank;
    logic          axis_fire;

    assign target_bank      = phase_active ? ~phase_sel : phase_sel;
    assign active_fill_bank = loading ? fill_bank_reg : target_bank;

    // Lock the selected bank for the whole input frame.
    assign s_axis_s2mm_tready = loading ? 1'b1 : !bank_full[target_bank];
    assign axis_fire          = s_axis_s2mm_tvalid && s_axis_s2mm_tready;

    assign wr_bank_sel = active_fill_bank;
    assign wr_en       = axis_fire;
    assign wr_addr     = wr_ptr;

    logic signed [AXIS_DATA_W-1:0] axis_word;
    assign axis_word = $signed(s_axis_s2mm_tdata);

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
            bank_full     <= 2'b00;
            loading       <= 1'b0;
            fill_bank_reg <= 1'b0;
            wr_ptr        <= '0;
        end else begin
            if (consume_pulse) begin
                bank_full[~phase_sel] <= 1'b0;
            end

            if (axis_fire) begin
                if (!loading) begin
                    fill_bank_reg <= target_bank;
                    loading       <= 1'b1;
                end

                if ($unsigned(wr_ptr) == (INPUT_SIZE-1)) begin
                    bank_full[active_fill_bank] <= 1'b1;
                    loading                     <= 1'b0;
                    wr_ptr                      <= '0;
                end else begin
                    wr_ptr <= wr_ptr + AW'(1);
                end
            end
        end
    end

    wire _unused_ok = &{1'b0, s_axis_s2mm_tkeep, s_axis_s2mm_tlast};

endmodule

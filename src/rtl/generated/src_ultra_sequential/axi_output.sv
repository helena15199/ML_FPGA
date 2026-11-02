// -----------------------------------------------------------------------------
// axi_output.sv
// Sends the final MLP output vector over AXI-Stream as signed AXIS_DATA_W-bit
// words. The external stream protocol intentionally mirrors the UART version:
//   - one output frame = exactly OUTPUT_SIZE words
//   - each external word is 16 bits by default
//   - internal ACT_W-bit values are sign-extended to AXIS_DATA_W bits
//   - tlast is asserted on the last output word
//
// This module assumes ACT_W <= AXIS_DATA_W.
// -----------------------------------------------------------------------------

module axi_output #(
    parameter int ACT_W       = 16,
    parameter int OUTPUT_SIZE = 64,
    parameter int AW          = (OUTPUT_SIZE > 1) ? $clog2(OUTPUT_SIZE) : 1,
    parameter logic FINAL_BANK_SEL = 1'b0,
    parameter int AXIS_DATA_W = 16,
    parameter int AXIS_KEEP_W = (AXIS_DATA_W + 7) / 8
)(
    input  logic                       clk,
    input  logic                       rstn,

    // Pulse from the MLP engine when the last layer has written
    // the final output vector into FINAL_BANK_SEL.
    input  logic                       produce_pulse,
    output logic                       busy,
    output logic                       done_pulse,

    // Read interface to the vector RAM bank.
    output logic                       rd_bank_sel,
    output logic                       rd_en,
    output logic [AW-1:0]              rd_addr,
    input  logic signed [ACT_W-1:0]    rd_data,

    output logic [AXIS_DATA_W-1:0]     m_axis_mm2s_tdata,
    output logic [AXIS_KEEP_W-1:0]     m_axis_mm2s_tkeep,
    output logic                       m_axis_mm2s_tlast,
    input  logic                       m_axis_mm2s_tready,
    output logic                       m_axis_mm2s_tvalid
);

    typedef enum logic [1:0] {
        S_IDLE,
        S_FETCH,
        S_SEND
    } state_t;

    state_t state;

    logic                         result_full;
    logic [AW-1:0]                 send_ptr;
    logic signed [ACT_W-1:0]       data_reg;
    logic signed [AXIS_DATA_W-1:0] axis_word;
    logic                         last_reg;
    logic                         axis_fire;

    assign axis_word = $signed(data_reg);

    assign axis_fire = m_axis_mm2s_tvalid && m_axis_mm2s_tready;
    assign busy      = result_full || (state != S_IDLE);

    always_comb begin
        rd_en       = 1'b0;
        rd_bank_sel = FINAL_BANK_SEL;
        rd_addr     = send_ptr;

        m_axis_mm2s_tdata  = axis_word;
        m_axis_mm2s_tkeep  = {AXIS_KEEP_W{1'b1}};
        m_axis_mm2s_tlast  = last_reg;
        m_axis_mm2s_tvalid = (state == S_SEND);

        unique case (state)
            S_IDLE: begin
                if (result_full) begin
                    rd_en   = 1'b1;
                    rd_addr = '0;
                end
            end

            S_FETCH: begin
                rd_addr = send_ptr;
            end

            S_SEND: begin
                if (axis_fire && !last_reg) begin
                    rd_en   = 1'b1;
                    rd_addr = send_ptr + AW'(1);
                end
            end

            default: begin
            end
        endcase
    end

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state       <= S_IDLE;
            result_full <= 1'b0;
            send_ptr    <= '0;
            data_reg    <= '0;
            last_reg    <= 1'b0;
            done_pulse  <= 1'b0;
        end else begin
            done_pulse <= 1'b0;

            if (produce_pulse) begin
                result_full <= 1'b1;
            end

            unique case (state)
                S_IDLE: begin
                    send_ptr <= '0;
                    if (result_full) begin
                        state <= S_FETCH;
                    end
                end

                // The vector RAM is synchronous: rd_en was asserted on the
                // previous cycle, so rd_data is captured here.
                S_FETCH: begin
                    data_reg <= rd_data;
                    last_reg <= ($unsigned(send_ptr) == (OUTPUT_SIZE-1));
                    state    <= S_SEND;
                end

                S_SEND: begin
                    if (axis_fire) begin
                        if (last_reg) begin
                            result_full <= 1'b0;
                            done_pulse  <= 1'b1;
                            state       <= S_IDLE;
                        end else begin
                            send_ptr <= send_ptr + AW'(1);
                            state    <= S_FETCH;
                        end
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

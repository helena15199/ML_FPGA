// -----------------------------------------------------------------------------
// axi_output.sv
// Sends output frames over AXI-Stream from the output boundary ping-pong buffer.
//
// Internal bank convention:
//   - during a phase, the last layer writes into ~phase_sel.
//   - out_produce_pulse is observed after phase_sel has toggled.
//   - therefore the bank that has just been produced is phase_sel when
//     produce_pulse is observed in this module.
//
// AXI convention:
//   - one output frame is exactly OUTPUT_SIZE signed ACT_W-bit words
//   - tlast is asserted on the last word of each output frame
// -----------------------------------------------------------------------------

module axi_output #(
    parameter int ACT_W       = 16,
    parameter int OUTPUT_SIZE = 64,
    parameter int AW          = (OUTPUT_SIZE > 1) ? $clog2(OUTPUT_SIZE) : 1,
    parameter int AXIS_DATA_W = 16,
    parameter int AXIS_KEEP_W = (AXIS_DATA_W + 7) / 8
)(
    input  logic                 clk,
    input  logic                 rstn,

    input  logic                 phase_sel,
    input  logic                 phase_active,
    input  logic                 produce_pulse,
    output logic [1:0]           bank_full,

    output logic                 rd_bank_sel,
    output logic                 rd_en,
    output logic [AW-1:0]        rd_addr,
    input  logic signed [ACT_W-1:0] rd_data,

    output logic [AXIS_DATA_W-1:0]        m_axis_mm2s_tdata,
    output logic [AXIS_KEEP_W-1:0]           m_axis_mm2s_tkeep,
    output logic                 m_axis_mm2s_tlast,
    input  logic                 m_axis_mm2s_tready,
    output logic                 m_axis_mm2s_tvalid
);

    typedef enum logic [1:0] {
        S_IDLE,
        S_FETCH,
        S_SEND
    } state_t;

    state_t state;

    logic                 send_bank_reg;
    logic [AW-1:0]        send_ptr;
    logic signed [ACT_W-1:0] data_reg;
    logic                 last_reg;

    logic signed [AXIS_DATA_W-1:0] axis_word;
    logic                 pick_bank;
    logic                 have_bank_to_send;
    logic                 axis_fire;

    assign axis_word = $signed(data_reg);

    assign axis_fire = m_axis_mm2s_tvalid && m_axis_mm2s_tready;

    // While a phase is active, ~phase_sel is potentially being written by the
    // pipeline, so only phase_sel is safe to read. Outside an active phase,
    // prefer ~phase_sel first because it is the bank that could block the next
    // internal output write.
    always_comb begin
        if (phase_active) begin
            have_bank_to_send = bank_full[phase_sel];
            pick_bank         = phase_sel;
        end else begin
            have_bank_to_send = bank_full[~phase_sel] || bank_full[phase_sel];
            pick_bank         = bank_full[~phase_sel] ? ~phase_sel : phase_sel;
        end
    end

    always_comb begin
        rd_en       = 1'b0;
        rd_addr     = send_ptr;
        rd_bank_sel = send_bank_reg;

        m_axis_mm2s_tdata  = axis_word;
        m_axis_mm2s_tkeep  = 2'b11;
        m_axis_mm2s_tlast  = last_reg;
        m_axis_mm2s_tvalid = (state == S_SEND);

        unique case (state)
            S_IDLE: begin
                if (have_bank_to_send) begin
                    rd_en       = 1'b1;
                    rd_addr     = '0;
                    rd_bank_sel = pick_bank;
                end
            end

            S_FETCH: begin
                rd_bank_sel = send_bank_reg;
            end

            S_SEND: begin
                rd_bank_sel = send_bank_reg;
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
            bank_full     <= 2'b00;
            state         <= S_IDLE;
            send_bank_reg <= 1'b0;
            send_ptr      <= '0;
            data_reg      <= '0;
            last_reg      <= 1'b0;
        end else begin
            if (produce_pulse) begin
                bank_full[phase_sel] <= 1'b1;
            end

            unique case (state)
                S_IDLE: begin
                    send_ptr <= '0;
                    last_reg <= 1'b0;
                    if (have_bank_to_send) begin
                        send_bank_reg <= pick_bank;
                        state         <= S_FETCH;
                    end
                end

                // pp_buffer is synchronous: rd_en was asserted in the previous cycle.
                S_FETCH: begin
                    data_reg <= rd_data;
                    last_reg <= ($unsigned(send_ptr) == (OUTPUT_SIZE-1));
                    state    <= S_SEND;
                end

                S_SEND: begin
                    if (axis_fire) begin
                        if (last_reg) begin
                            bank_full[send_bank_reg] <= 1'b0;
                            state                    <= S_IDLE;
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

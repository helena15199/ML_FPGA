// -----------------------------------------------------------------------------
// uart_output.sv
// Sends the final MLP output vector over UART as signed 16-bit words.
//
// uart_tx handles the UART byte protocol.
// uart_output only handles the application framing:
//   - one output frame = exactly OUTPUT_SIZE words
//   - one word = two UART bytes, LSB first
//   - internal ACT_W-bit values are sign-extended to 16 bits
//   - output RAM bank availability follows the same phase/bank_full interface
//
// This module assumes ACT_W <= 16.
// -----------------------------------------------------------------------------

module uart_output #(
    parameter int ACT_W       = 16,
    parameter int OUTPUT_SIZE = 64,
    parameter int AW          = (OUTPUT_SIZE > 1) ? $clog2(OUTPUT_SIZE) : 1
)(
    input  logic                 clk,
    input  logic                 rstn,

    // Same pipeline control interface.
    input  logic                 phase_sel,
    input  logic                 phase_active,
    input  logic                 produce_pulse,
    output logic [1:0]           bank_full,

    // Read interface to the vector RAM bank.
    output logic                 rd_bank_sel,
    output logic                 rd_en,
    output logic [AW-1:0]        rd_addr,
    input  logic signed [ACT_W-1:0] rd_data,

    // Byte interface to uart_tx.
    output logic [7:0]           tx_data,
    output logic                 tx_valid,
    input  logic                 tx_busy
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_FETCH,
        S_SEND_LSB,
        S_WAIT_LSB,
        S_SEND_MSB,
        S_WAIT_MSB,
        S_NEXT
    } state_t;

    state_t state;

    logic                 send_bank_reg;
    logic [AW-1:0]        send_ptr;
    logic signed [ACT_W-1:0] data_reg;
    logic signed [15:0]   tx_word;
    logic                 last_reg;

    logic                 have_bank_to_send;
    logic                 pick_bank;

    assign tx_word = $signed(data_reg);

    // Same bank selection policy as the pipeline output side:
    // - during an active phase, avoid reading the bank currently being written
    //   by the last layer;
    // - outside a phase, send any full bank, prioritizing the one that may block
    //   the next producer write.
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
        rd_bank_sel = send_bank_reg;
        rd_addr     = send_ptr;
        tx_data     = 8'h00;
        tx_valid    = 1'b0;

        unique case (state)
            S_IDLE: begin
                if (have_bank_to_send) begin
                    rd_en       = 1'b1;
                    rd_addr     = '0;
                    rd_bank_sel = pick_bank;
                end
            end

            S_NEXT: begin
                rd_en       = 1'b1;
                rd_addr     = send_ptr + AW'(1);
                rd_bank_sel = send_bank_reg;
            end

            S_SEND_LSB: begin
                if (!tx_busy) begin
                    tx_data  = tx_word[7:0];
                    tx_valid = 1'b1;
                end
            end

            S_SEND_MSB: begin
                if (!tx_busy) begin
                    tx_data  = tx_word[15:8];
                    tx_valid = 1'b1;
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
                    if (have_bank_to_send) begin
                        send_bank_reg <= pick_bank;
                        state         <= S_FETCH;
                    end
                end

                // The vector RAM is synchronous: rd_en was asserted on the
                // previous cycle, so rd_data is captured here.
                S_FETCH: begin
                    data_reg <= rd_data;
                    last_reg <= ($unsigned(send_ptr) == (OUTPUT_SIZE-1));
                    state    <= S_SEND_LSB;
                end

                S_SEND_LSB: begin
                    if (!tx_busy) begin
                        state <= S_WAIT_LSB;
                    end
                end

                S_WAIT_LSB: begin
                    if (!tx_busy) begin
                        state <= S_SEND_MSB;
                    end
                end

                S_SEND_MSB: begin
                    if (!tx_busy) begin
                        state <= S_WAIT_MSB;
                    end
                end

                S_WAIT_MSB: begin
                    if (!tx_busy) begin
                        if (last_reg) begin
                            bank_full[send_bank_reg] <= 1'b0;
                            state                    <= S_IDLE;
                        end else begin
                            state <= S_NEXT;
                        end
                    end
                end

                S_NEXT: begin
                    send_ptr <= send_ptr + AW'(1);
                    state    <= S_FETCH;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

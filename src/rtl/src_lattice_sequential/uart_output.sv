// -----------------------------------------------------------------------------
// uart_output.sv
// Sends the final MLP output vector over UART as signed 16-bit words.
// Values are read from the selected final RAM bank and transmitted
// little-endian, one word at a time.
// This module assumes ACT_W <= 16.
// -----------------------------------------------------------------------------

module uart_output #(
    parameter int ACT_W          = 16,
    parameter int OUTPUT_SIZE = 64,
    parameter int AW          = (OUTPUT_SIZE > 1) ? $clog2(OUTPUT_SIZE) : 1,
    parameter logic FINAL_BANK_SEL = 1'b0
)(
    input  logic                 clk,
    input  logic                 rstn,

    // Pulse from the MLP engine when the last layer has written
    // the final output vector into FINAL_BANK_SEL.
    input  logic                 produce_pulse,
    output logic                 busy,
    output logic                 done_pulse,

    // Read interface to the vector RAM bank.
    output logic                 rd_bank_sel,
    output logic                 rd_en,
    output logic [AW-1:0]        rd_addr,
    input  logic signed [ACT_W-1:0] rd_data,

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

    logic                    result_full;
    logic [AW-1:0]            send_ptr;
    logic signed [ACT_W-1:0]     data_reg;
    logic signed [15:0]       tx_word;
    logic                    last_reg;

    // Fixed UART output protocol: signed 16-bit words, little-endian.
    // If ACT_W is smaller than 16 bits, the value is sign-extended to 16 bits.
    assign tx_word = $signed(data_reg);

    assign busy = result_full || (state != S_IDLE);

    always_comb begin
        rd_en       = 1'b0;
        rd_bank_sel = FINAL_BANK_SEL;
        rd_addr     = send_ptr;
        tx_data     = 8'h00;
        tx_valid    = 1'b0;

        unique case (state)
            S_IDLE: begin
                if (result_full) begin
                    rd_en   = 1'b1;
                    rd_addr = '0;
                end
            end

            S_NEXT: begin
                rd_en   = 1'b1;
                rd_addr = send_ptr + AW'(1);
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

                // The RAM is synchronous: rd_en was asserted on the previous cycle.
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
                            result_full <= 1'b0;
                            done_pulse  <= 1'b1;
                            state       <= S_IDLE;
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

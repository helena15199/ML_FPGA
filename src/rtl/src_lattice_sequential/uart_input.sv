// -----------------------------------------------------------------------------
// uart_input.sv
// Receives an input vector over UART as signed 16-bit words.
// Values are received little-endian, resized to ACT_W bits, and written
// into RAM bank A, one word at a time.
// This module assumes ACT_W <= 16.
// -----------------------------------------------------------------------------

module uart_input #(
    parameter int ACT_W         = 16,
    parameter int INPUT_SIZE = 64,
    parameter int AW         = (INPUT_SIZE > 1) ? $clog2(INPUT_SIZE) : 1,
    parameter int IDLE_TIMEOUT_CYCLES = 120_000 // 12 MHz, 120_000 cycles = 10 ms.
)(
    input  logic                 clk,
    input  logic                 rstn,

    input  logic [7:0]           uart_data,
    input  logic                 uart_valid,

    // High only when RAM banks A/B are available.
    input  logic                 accept_enable,

    // Pulse from the MLP engine when the frame loaded into RAM bank A is consumed.
    input  logic                 consume_pulse,
    output logic                 frame_full,
    output logic                 loading,

    // Write interface to the vector RAM bank.
    // The input frame is always loaded into bank A, so wr_bank_sel = 0.
    output logic                 wr_bank_sel,
    output logic                 wr_en,
    output logic [AW-1:0]        wr_addr,
    output logic signed [ACT_W-1:0] wr_data
);

    localparam int TIMEOUT_W = (IDLE_TIMEOUT_CYCLES > 1) ? $clog2(IDLE_TIMEOUT_CYCLES + 1) : 1;

    logic [AW-1:0]          wr_ptr;
    logic [7:0]             lsb_byte;
    logic                   byte_phase; // 0: wait for LSB, 1: wait for MSB
    logic                   byte_fire;
    logic                   word_fire;
    logic [TIMEOUT_W-1:0]   idle_cnt;

    logic signed [15:0]      uart_word;

    assign uart_word = $signed({uart_data, lsb_byte});

    assign byte_fire = uart_valid && accept_enable && !frame_full;
    assign word_fire = byte_fire && byte_phase;

    assign wr_bank_sel = 1'b0; // bank A
    assign wr_en       = word_fire;
    assign wr_addr     = wr_ptr;

    // Fixed UART input protocol: signed 16-bit words, little-endian.
    // If ACT_W is smaller than 16 bits, only the ACT_W least significant bits are kept.
    // The host is expected to pack the useful payload in the LSBs.
    generate
        if (ACT_W == 16) begin : g_uart_rx_same_width
            assign wr_data = uart_word;
        end else begin : g_uart_rx_truncate
            assign wr_data = $signed(uart_word[ACT_W-1:0]);
        end
    endgenerate

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            frame_full <= 1'b0;
            loading    <= 1'b0;
            wr_ptr     <= '0;
            lsb_byte   <= '0;
            byte_phase <= 1'b0;
            idle_cnt   <= '0;
        end else begin
            if (consume_pulse) begin
                frame_full <= 1'b0;
            end

            if (byte_fire) begin
                idle_cnt <= '0;

                if (!byte_phase) begin
                    // First byte of the 16-bit word: LSB.
                    lsb_byte   <= uart_data;
                    byte_phase <= 1'b1;
                    loading    <= 1'b1;
                end else begin
                    // Second byte of the 16-bit word: MSB.
                    byte_phase <= 1'b0;

                    if ($unsigned(wr_ptr) == (INPUT_SIZE-1)) begin
                        frame_full <= 1'b1;
                        loading    <= 1'b0;
                        wr_ptr     <= '0;
                    end else begin
                        wr_ptr  <= wr_ptr + AW'(1);
                        loading <= 1'b1;
                    end
                end
            end else if (loading && !frame_full) begin
                // A frame has started but is not complete.
                // If no byte is received before the timeout, discard the partial frame.
                if ($unsigned(idle_cnt) >= (IDLE_TIMEOUT_CYCLES-1)) begin
                    loading    <= 1'b0;
                    wr_ptr     <= '0;
                    lsb_byte   <= '0;
                    byte_phase <= 1'b0;
                    idle_cnt   <= '0;
                end else begin
                    idle_cnt <= idle_cnt + TIMEOUT_W'(1);
                end
            end else begin
                idle_cnt <= '0;
            end
        end
    end

endmodule

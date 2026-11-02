// -----------------------------------------------------------------------------
// uart_rx.sv
// Simple UART receiver with 8 data bits, no parity, and one stop bit.
// The asynchronous RX input is synchronized before being sampled.
// Clock frequency and baud rate are parameterizable.
// -----------------------------------------------------------------------------

module uart_rx #(
    parameter int CLK_FREQ  = 12_000_000,
    parameter int BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rstn,

    input  logic rx,

    output logic [7:0] data_out,
    output logic       data_valid
);

    localparam int CLKS_PER_BIT = (CLK_FREQ + (BAUD_RATE/2)) / BAUD_RATE;
    localparam int CNT_W        = (CLKS_PER_BIT > 1) ? $clog2(CLKS_PER_BIT) : 1;

    typedef enum logic [1:0] {
        S_IDLE,
        S_START,
        S_DATA,
        S_STOP
    } state_t;

    state_t state;

    logic [CNT_W-1:0] clk_cnt;
    logic [2:0]       bit_idx;
    logic [7:0]       data_shift;

    // Synchronize the external serial input into the clk domain.    
    logic rx_meta;
    logic rx_sync;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state      <= S_IDLE;
            clk_cnt    <= '0;
            bit_idx    <= '0;
            data_shift <= '0;
            data_out   <= '0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;

            unique case (state)
                S_IDLE: begin
                    clk_cnt <= '0;
                    bit_idx <= '0;

                    // UART idle line is high. A start bit is detected as a low level.
                    if (!rx_sync) begin
                        state <= S_START;
                    end
                end

                S_START: begin
                    // Validate the start bit at its midpoint.
                    if (clk_cnt == (CLKS_PER_BIT/2)) begin
                        clk_cnt <= '0;

                        if (!rx_sync) begin
                            state <= S_DATA;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + CNT_W'(1);
                    end
                end

                S_DATA: begin
                    if (clk_cnt == (CLKS_PER_BIT-1)) begin
                        clk_cnt             <= '0;
                        data_shift[bit_idx] <= rx_sync;

                        if (bit_idx == 3'd7) begin
                            bit_idx <= '0;
                            state   <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + CNT_W'(1);
                    end
                end

                S_STOP: begin
                    if (clk_cnt == (CLKS_PER_BIT-1)) begin
                        clk_cnt <= '0;
                        state   <= S_IDLE;

                        // Accept the frame only if the stop bit is high.
                        if (rx_sync) begin
                            data_out   <= data_shift;
                            data_valid <= 1'b1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + CNT_W'(1);
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

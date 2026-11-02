// -----------------------------------------------------------------------------
// uart_tx.sv
// Simple UART transmitter with 8 data bits, no parity, and one stop bit.
// The frame is serialized LSB first when data_valid is asserted.
// Clock frequency and baud rate are parameterizable.
// -----------------------------------------------------------------------------

module uart_tx #(
    parameter int CLK_FREQ  = 12_000_000,
    parameter int BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rstn,

    input  logic [7:0] data_in,
    input  logic       data_valid,

    output logic tx,
    output logic busy
);

    localparam int CLKS_PER_BIT = (CLK_FREQ + (BAUD_RATE/2)) / BAUD_RATE;
    localparam int CNT_W        = (CLKS_PER_BIT > 1) ? $clog2(CLKS_PER_BIT) : 1;

    typedef enum logic {
        S_IDLE,
        S_SEND
    } state_t;

    state_t state;

    logic [CNT_W-1:0] clk_cnt;
    logic [3:0]       bit_idx;
    logic [9:0]       frame_shift; // {stop bit, data[7:0], start bit}

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state       <= S_IDLE;
            clk_cnt     <= '0;
            bit_idx     <= '0;
            frame_shift <= 10'h3FF;
            tx          <= 1'b1;
            busy        <= 1'b0;
        end else begin
            unique case (state)
                S_IDLE: begin
                    tx      <= 1'b1;
                    busy    <= 1'b0;
                    clk_cnt <= '0;
                    bit_idx <= '0;

                    if (data_valid) begin
                        frame_shift <= {1'b1, data_in, 1'b0};
                        tx          <= 1'b0; // start bit
                        busy        <= 1'b1;
                        state       <= S_SEND;
                    end
                end

                S_SEND: begin
                    busy <= 1'b1;

                    if (clk_cnt == (CLKS_PER_BIT-1)) begin
                        clk_cnt <= '0;

                        if (bit_idx == 4'd9) begin
                            bit_idx <= '0;
                            tx      <= 1'b1;
                            busy    <= 1'b0;
                            state   <= S_IDLE;
                        end else begin
                            bit_idx <= bit_idx + 4'd1;
                            tx      <= frame_shift[bit_idx + 4'd1];
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

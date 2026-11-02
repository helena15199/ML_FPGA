`include "scales.vh"

// -----------------------------------------------------------------------------
// SOC_MLP.sv
// Top-level UART wrapper for the pipelined MLP core.
//
// Architecture:
//   UART RX -> uart_input -> input ping-pong buffer
//                         -> mlp_pipeline
//                         -> output ping-pong buffer -> uart_output -> UART TX
//
// uart_rx and uart_tx only implement the byte-level UART protocol.
// uart_input and uart_output implement the application framing:
//   - one frame contains a fixed number of signed 16-bit words
//   - words are transmitted little-endian: LSB byte first, then MSB byte
//
// Model dimensions and fixed-point scales come from scales.vh.
// A few structural values are filled by Python during RTL generation.
// -----------------------------------------------------------------------------

module SOC_MLP #(
    parameter int CLK_FREQ       = 6_000_000,
    parameter int BAUD_RATE      = 115200,
    parameter bit LED_ACTIVE_LOW = 1'b0
)(
    input  logic uart_rx,
    output logic uart_tx
);

    // -------------------------------------------------------------------------
    // Internal Lattice clock.
    // -------------------------------------------------------------------------
    logic clk;

    HSOSC #(
        .CLKHF_DIV("0b11")   // 6 MHz
    ) osc_inst (
        .CLKHFPU(1'b1),
        .CLKHFEN(1'b1),
        .CLKHF  (clk)
    );

    // -------------------------------------------------------------------------
    // Internal power-on reset.
    // No external reset button is required.
    // 24 bits gives about 2.8 s at 6 MHz.
    // -------------------------------------------------------------------------
    logic        rstn    = 1'b0;
    logic [23:0] por_cnt = 24'd0;

    always_ff @(posedge clk) begin
        if (por_cnt != 24'hFF_FFFF) begin
            por_cnt <= por_cnt + 24'd1;
            rstn    <= 1'b0;
        end else begin
            rstn <= 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Model dimensions.
    // -------------------------------------------------------------------------
    localparam int ACT_W_LOCAL  = `ACT_W;
    localparam int DSP_PW_LOCAL = `DSP_PW;
    localparam int INPUT_SZ     = `N0;
    localparam int OUTPUT_SZ    = {output_size_expr};

    localparam int AW_IN     = (INPUT_SZ  > 1) ? $clog2(INPUT_SZ ) : 1;
    localparam int AW_OUT    = (OUTPUT_SZ > 1) ? $clog2(OUTPUT_SZ) : 1;

    // -------------------------------------------------------------------------
    // UART byte interfaces.
    // -------------------------------------------------------------------------
    logic [7:0] rx_data;
    logic       rx_valid;

    logic [7:0] tx_data;
    logic       tx_valid;
    logic       tx_busy;

    // -------------------------------------------------------------------------
    // Pipeline boundary bank status and phase control.
    // -------------------------------------------------------------------------
    logic [1:0] in_bank_full;
    logic [1:0] out_bank_full;

    logic       phase_sel;
    logic       phase_active;
    logic       in_consume_pulse;
    logic       out_produce_pulse;

    // -------------------------------------------------------------------------
    // Input ping-pong buffer ports.
    // -------------------------------------------------------------------------
    logic                 in_pp_wr_bank_sel;
    logic                 in_pp_wr_en;
    logic [AW_IN-1:0]     in_pp_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] in_pp_wr_data;

    logic                 in_pp_rd_en;
    logic [AW_IN-1:0]     in_pp_rd_addr;
    logic signed [ACT_W_LOCAL-1:0] in_pp_rd_data;

    logic                 input_loading;

    // -------------------------------------------------------------------------
    // Output ping-pong buffer ports.
    // -------------------------------------------------------------------------
    logic                 out_pp_wr_en;
    logic [AW_OUT-1:0]    out_pp_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] out_pp_wr_data;

    logic                 out_pp_rd_bank_sel;
    logic                 out_pp_rd_en;
    logic [AW_OUT-1:0]    out_pp_rd_addr;
    logic signed [ACT_W_LOCAL-1:0] out_pp_rd_data;

    // -------------------------------------------------------------------------
    // UART RX.
    // -------------------------------------------------------------------------
    uart_rx #(
        .CLK_FREQ (CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart_rx (
        .clk       (clk),
        .rstn      (rstn),
        .rx        (uart_rx),
        .data_out  (rx_data),
        .data_valid(rx_valid)
    );

    // -------------------------------------------------------------------------
    // UART input framing.
    // Loads complete input frames into one free input ping-pong bank.
    // -------------------------------------------------------------------------
    uart_input #(
        .ACT_W     (ACT_W_LOCAL),
        .INPUT_SIZE(INPUT_SZ),
        .AW        (AW_IN)
    ) u_uart_input (
        .clk          (clk),
        .rstn         (rstn),

        .rx_data      (rx_data),
        .rx_valid     (rx_valid),

        .phase_sel    (phase_sel),
        .phase_active (phase_active),
        .consume_pulse(in_consume_pulse),
        .bank_full    (in_bank_full),
        .loading      (input_loading),

        .wr_bank_sel  (in_pp_wr_bank_sel),
        .wr_en        (in_pp_wr_en),
        .wr_addr      (in_pp_wr_addr),
        .wr_data      (in_pp_wr_data)
    );

    // -------------------------------------------------------------------------
    // Input boundary ping-pong buffer.
    // The MLP consumes phase_sel while uart_input fills a free bank.
    // -------------------------------------------------------------------------
    pp_buffer #(
        .ACT_W(ACT_W_LOCAL),
        .DEPTH(INPUT_SZ),
        .AW   (AW_IN)
    ) u_input_pp (
        .clk        (clk),

        .rd_bank_sel(phase_sel),
        .rd_en      (in_pp_rd_en),
        .rd_addr    (in_pp_rd_addr),
        .rd_data    (in_pp_rd_data),

        .wr_bank_sel(in_pp_wr_bank_sel),
        .wr_en      (in_pp_wr_en),
        .wr_addr    (in_pp_wr_addr),
        .wr_data    (in_pp_wr_data)
    );

    // -------------------------------------------------------------------------
    // Pipelined MLP core.
    // -------------------------------------------------------------------------
    mlp_pipeline #(
        .ACT_W      (ACT_W_LOCAL),
        .WEIGHT_W   (`WEIGHT_W),
        .BIAS_W     (`BIAS_W),
        .DSP_PW     (DSP_PW_LOCAL),
        .INPUT_SIZE (INPUT_SZ),
        .OUTPUT_SIZE(OUTPUT_SZ)
    ) u_mlp_pipeline (
        .clk              (clk),
        .rstn             (rstn),

        .in_bank_full     (in_bank_full),
        .out_bank_full    (out_bank_full),

        .phase_sel        (phase_sel),
        .phase_active     (phase_active),
        .in_consume_pulse (in_consume_pulse),
        .out_produce_pulse(out_produce_pulse),

        .in_rd_en         (in_pp_rd_en),
        .in_rd_addr       (in_pp_rd_addr),
        .in_rd_data       (in_pp_rd_data),

        .out_wr_en        (out_pp_wr_en),
        .out_wr_addr      (out_pp_wr_addr),
        .out_wr_data      (out_pp_wr_data)
    );

    // -------------------------------------------------------------------------
    // Output boundary ping-pong buffer.
    // For each active phase, the last layer writes into ~phase_sel.
    // -------------------------------------------------------------------------
    pp_buffer #(
        .ACT_W(ACT_W_LOCAL),
        .DEPTH(OUTPUT_SZ),
        .AW   (AW_OUT)
    ) u_output_pp (
        .clk        (clk),

        .rd_bank_sel(out_pp_rd_bank_sel),
        .rd_en      (out_pp_rd_en),
        .rd_addr    (out_pp_rd_addr),
        .rd_data    (out_pp_rd_data),

        .wr_bank_sel(~phase_sel),
        .wr_en      (out_pp_wr_en),
        .wr_addr    (out_pp_wr_addr),
        .wr_data    (out_pp_wr_data)
    );

    // -------------------------------------------------------------------------
    // UART output framing.
    // Sends complete output frames from full output ping-pong banks.
    //
    // Important convention:
    //   phase_sel  = input bank consumed by the current phase
    //   ~phase_sel = output bank produced by the current phase
    // Therefore uart_output must mark bank_full[~phase_sel] on produce_pulse.
    // -------------------------------------------------------------------------
    uart_output #(
        .ACT_W      (ACT_W_LOCAL),
        .OUTPUT_SIZE(OUTPUT_SZ),
        .AW         (AW_OUT)
    ) u_uart_output (
        .clk          (clk),
        .rstn         (rstn),

        .phase_sel    (phase_sel),
        .phase_active (phase_active),
        .produce_pulse(out_produce_pulse),
        .bank_full    (out_bank_full),

        .rd_bank_sel  (out_pp_rd_bank_sel),
        .rd_en        (out_pp_rd_en),
        .rd_addr      (out_pp_rd_addr),
        .rd_data      (out_pp_rd_data),

        .tx_data      (tx_data),
        .tx_valid     (tx_valid),
        .tx_busy      (tx_busy)
    );

    // -------------------------------------------------------------------------
    // UART TX.
    // -------------------------------------------------------------------------
    uart_tx #(
        .CLK_FREQ (CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart_tx (
        .clk       (clk),
        .rstn      (rstn),

        .data_in   (tx_data),
        .data_valid(tx_valid),

        .tx        (uart_tx),
        .busy      (tx_busy)
    );

    // Keep intentionally unused debug/status signals visible to synthesis.
    wire _unused_ok = &{1'b0, input_loading, LED_ACTIVE_LOW};

endmodule

`include "scales.vh"

module SOC_MLP #(
    parameter int CLK_FREQ       = 6_000_000,
    parameter int BAUD_RATE      = 115200,
    parameter int UART_IDLE_TIMEOUT_CYCLES = CLK_FREQ / 100, // 10 ms timeout
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
            rstn    <= 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Internal dimensions.
    // -------------------------------------------------------------------------
    localparam int ACT_W_LOCAL    = `ACT_W;
    localparam int DSP_PW_LOCAL   = `DSP_PW;
    localparam int INPUT_SZ       = `N0;
    localparam int OUTPUT_SZ      = {output_size_expr};
    localparam int NUM_LINEAR_CFG = `NUM_LINEAR;
    localparam int MAX_NEURONS    = {max_neurons_expr};
    localparam int VEC_AW         = (MAX_NEURONS > 1) ? $clog2(MAX_NEURONS) : 1;

    // Ping-pong RAM for the real MLP:
    // layer 0 writes B, layer 1 writes A, layer 2 writes B, ...
    // therefore the final bank is 1 if NUM_LINEAR is odd, 0 if it is even.
    // With NUM_LINEAR=6, the final output is in MEM_A => FINAL_BANK_SEL=0.
    localparam logic FINAL_BANK_SEL = ((NUM_LINEAR_CFG % 2) == 1) ? 1'b1 : 1'b0;

    // -------------------------------------------------------------------------
    // UART RX/TX bytes.
    // -------------------------------------------------------------------------
    logic [7:0] rx_data;
    logic       rx_valid;

    logic [7:0] tx_data;
    logic       tx_valid;
    logic       tx_busy;

    // -------------------------------------------------------------------------
    // Synchronization.
    // -------------------------------------------------------------------------
    logic input_accept_enable;
    logic input_frame_full;
    logic input_loading;
    logic input_consume_pulse;

    logic output_produce_pulse;
    logic output_busy;
    logic output_done_pulse;

    logic mlp_busy;
    logic mlp_done_pulse;

    // -------------------------------------------------------------------------
    // RAM ports driven by uart_input.
    // -------------------------------------------------------------------------
    logic                      uart_in_wr_bank_sel;
    logic                      uart_in_wr_en;
    logic [VEC_AW-1:0]         uart_in_wr_addr;
    logic signed [ACT_W_LOCAL-1:0]      uart_in_wr_data;

    // -------------------------------------------------------------------------
    // RAM ports driven by the sequential MLP engine.
    // -------------------------------------------------------------------------
    logic                      mlp_vec_rd_en;
    logic                      mlp_vec_rd_bank_sel;
    logic [VEC_AW-1:0]         mlp_vec_rd_addr;

    logic                      mlp_vec_wr_en;
    logic                      mlp_vec_wr_bank_sel;
    logic [VEC_AW-1:0]         mlp_vec_wr_addr;
    logic signed [ACT_W_LOCAL-1:0]      mlp_vec_wr_data;

    // -------------------------------------------------------------------------
    // RAM ports driven by uart_output.
    // -------------------------------------------------------------------------
    logic                      uart_out_rd_bank_sel;
    logic                      uart_out_rd_en;
    logic [VEC_AW-1:0]         uart_out_rd_addr;

    // -------------------------------------------------------------------------
    // MEM_A / MEM_B arbitration.
    // uart_input writes A when the MLP is not running.
    // mlp_sequential reads/writes A/B while it is busy.
    // uart_output reads the final bank when the MLP is done.
    // -------------------------------------------------------------------------
    logic                      ram_rd_en;
    logic                      ram_rd_bank_sel;
    logic [VEC_AW-1:0]         ram_rd_addr;
    logic signed [ACT_W_LOCAL-1:0]      ram_rd_data;

    logic                      ram_wr_en;
    logic                      ram_wr_bank_sel;
    logic [VEC_AW-1:0]         ram_wr_addr;
    logic signed [ACT_W_LOCAL-1:0]      ram_wr_data;

    logic                      mem_a_rd_en;
    logic                      mem_b_rd_en;
    logic                      mem_a_wr_en;
    logic                      mem_b_wr_en;
    logic signed [ACT_W_LOCAL-1:0]      mem_a_rd_data;
    logic signed [ACT_W_LOCAL-1:0]      mem_b_rd_data;
    logic                      ram_rd_bank_sel_q;

    assign input_accept_enable = !mlp_busy && !output_busy;

    assign ram_rd_en       = mlp_busy ? mlp_vec_rd_en       : uart_out_rd_en;
    assign ram_rd_bank_sel = mlp_busy ? mlp_vec_rd_bank_sel : uart_out_rd_bank_sel;
    assign ram_rd_addr     = mlp_busy ? mlp_vec_rd_addr     : uart_out_rd_addr;

    assign ram_wr_en       = mlp_busy ? mlp_vec_wr_en       : uart_in_wr_en;
    assign ram_wr_bank_sel = mlp_busy ? mlp_vec_wr_bank_sel : uart_in_wr_bank_sel;
    assign ram_wr_addr     = mlp_busy ? mlp_vec_wr_addr     : uart_in_wr_addr;
    assign ram_wr_data     = mlp_busy ? mlp_vec_wr_data     : uart_in_wr_data;

    assign mem_a_rd_en = ram_rd_en && !ram_rd_bank_sel;
    assign mem_b_rd_en = ram_rd_en &&  ram_rd_bank_sel;
    assign mem_a_wr_en = ram_wr_en && !ram_wr_bank_sel;
    assign mem_b_wr_en = ram_wr_en &&  ram_wr_bank_sel;

    // Synchronous RAM: store the requested bank during the rd_en cycle.
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            ram_rd_bank_sel_q <= 1'b0;
        end else if (ram_rd_en) begin
            ram_rd_bank_sel_q <= ram_rd_bank_sel;
        end
    end

    assign ram_rd_data = ram_rd_bank_sel_q ? mem_b_rd_data : mem_a_rd_data;

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
    // UART input: assembles N0 int16 LSB/MSB words and loads MEM_A.
    // Timeout version used to ignore a partial frame after reconnection.
    // -------------------------------------------------------------------------
    uart_input #(
        .ACT_W                 (ACT_W_LOCAL),
        .INPUT_SIZE         (INPUT_SZ),
        .AW                 (VEC_AW),
        .IDLE_TIMEOUT_CYCLES(UART_IDLE_TIMEOUT_CYCLES)
    ) u_uart_input (
        .clk          (clk),
        .rstn         (rstn),
        .uart_data    (rx_data),
        .uart_valid   (rx_valid),
        .accept_enable(input_accept_enable),
        .consume_pulse(input_consume_pulse),
        .frame_full   (input_frame_full),
        .loading      (input_loading),
        .wr_bank_sel  (uart_in_wr_bank_sel),
        .wr_en        (uart_in_wr_en),
        .wr_addr      (uart_in_wr_addr),
        .wr_data      (uart_in_wr_data)
    );

    mem_a #(
        .ACT_W(ACT_W_LOCAL),
        .DEPTH(MAX_NEURONS),
        .AW   (VEC_AW)
    ) u_mem_a (
        .clk    (clk),
        .rd_en  (mem_a_rd_en),
        .rd_addr(ram_rd_addr),
        .rd_data(mem_a_rd_data),
        .wr_en  (mem_a_wr_en),
        .wr_addr(ram_wr_addr),
        .wr_data(ram_wr_data)
    );

    // -------------------------------------------------------------------------
    // Complete sequential MLP model.
    // -------------------------------------------------------------------------
    mlp_sequential #(
        .ACT_W         (ACT_W_LOCAL),
        .WEIGHT_W      (`WEIGHT_W),
        .BIAS_W        (`BIAS_W),
        .DSP_PW        (DSP_PW_LOCAL),
        .INPUT_SIZE    (INPUT_SZ),
        .OUTPUT_SIZE   (OUTPUT_SZ),
        .NUM_LINEAR_CFG(NUM_LINEAR_CFG),
        .MAX_NEURONS   (MAX_NEURONS),
        .VEC_AW        (VEC_AW)
    ) u_mlp_seq (
        .clk                 (clk),
        .rstn                (rstn),
        .input_frame_full    (input_frame_full),
        .output_busy         (output_busy),
        .input_consume_pulse (input_consume_pulse),
        .output_produce_pulse(output_produce_pulse),
        .vec_rd_en           (mlp_vec_rd_en),
        .vec_rd_bank_sel     (mlp_vec_rd_bank_sel),
        .vec_rd_addr         (mlp_vec_rd_addr),
        .vec_rd_data         (ram_rd_data),
        .vec_wr_en           (mlp_vec_wr_en),
        .vec_wr_bank_sel     (mlp_vec_wr_bank_sel),
        .vec_wr_addr         (mlp_vec_wr_addr),
        .vec_wr_data         (mlp_vec_wr_data),
        .busy                (mlp_busy),
        .done_pulse          (mlp_done_pulse)
    );

    mem_b #(
        .ACT_W(ACT_W_LOCAL),
        .DEPTH(MAX_NEURONS),
        .AW   (VEC_AW)
    ) u_mem_b (
        .clk    (clk),
        .rd_en  (mem_b_rd_en),
        .rd_addr(ram_rd_addr),
        .rd_data(mem_b_rd_data),
        .wr_en  (mem_b_wr_en),
        .wr_addr(ram_wr_addr),
        .wr_data(ram_wr_data)
    );

    // -------------------------------------------------------------------------
    // UART output: reads the final output vector from the MLP result bank.
    // -------------------------------------------------------------------------
    uart_output #(
        .ACT_W            (ACT_W_LOCAL),
        .OUTPUT_SIZE   (OUTPUT_SZ),
        .AW            (VEC_AW),
        .FINAL_BANK_SEL(FINAL_BANK_SEL)
    ) u_uart_output (
        .clk          (clk),
        .rstn         (rstn),
        .produce_pulse(output_produce_pulse),
        .busy         (output_busy),
        .done_pulse   (output_done_pulse),
        .rd_bank_sel  (uart_out_rd_bank_sel),
        .rd_en        (uart_out_rd_en),
        .rd_addr      (uart_out_rd_addr),
        .rd_data      (ram_rd_data),
        .tx_data      (tx_data),
        .tx_valid     (tx_valid),
        .tx_busy      (tx_busy)
    );

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

    wire _unused_ok = &{1'b0, input_loading, output_done_pulse, mlp_done_pulse};

endmodule

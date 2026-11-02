`include "scales.vh"

// -----------------------------------------------------------------------------
// SOC_MLP.sv
// Top-level AXI-Stream wrapper for the pipelined MLP core.
//
// Architecture:
//   AXI-Stream RX -> axi_input -> input ping-pong buffer
//                              -> mlp_pipeline
//                              -> output ping-pong buffer -> axi_output -> AXI-Stream TX
//
// This top is intended for the Ultra96/testbench flow. It keeps the same
// internal ping-pong and phase-control architecture as the UART pipeline top,
// but replaces uart_input/uart_output by axi_input/axi_output.
// -----------------------------------------------------------------------------

module SOC_MLP (
    input  logic                 clk,
    input  logic                 rstn,

    input  logic [15:0] s_axis_s2mm_tdata,
    input  logic [1:0]           s_axis_s2mm_tkeep,
    input  logic                 s_axis_s2mm_tlast,
    input  logic                 s_axis_s2mm_tvalid,
    output logic                 s_axis_s2mm_tready,

    output logic [15:0] m_axis_mm2s_tdata,
    output logic [1:0]           m_axis_mm2s_tkeep,
    output logic                 m_axis_mm2s_tlast,
    input  logic                 m_axis_mm2s_tready,
    output logic                 m_axis_mm2s_tvalid
);

    localparam int AXIS_DATA_W  = 16;
    localparam int AXIS_KEEP_W  = (AXIS_DATA_W + 7) / 8;
    localparam int ACT_W_LOCAL  = `ACT_W;
    localparam int DSP_PW_LOCAL = `DSP_PW;
    localparam int INPUT_SZ     = `N0;
    localparam int OUTPUT_SZ    = {output_size_expr};

    localparam int AW_IN     = (INPUT_SZ  > 1) ? $clog2(INPUT_SZ ) : 1;
    localparam int AW_OUT    = (OUTPUT_SZ > 1) ? $clog2(OUTPUT_SZ) : 1;

    logic [1:0] in_bank_full;
    logic [1:0] out_bank_full;

    logic phase_sel;
    logic phase_active;
    logic in_consume_pulse;
    logic out_produce_pulse;

    logic                 in_pp_wr_bank_sel;
    logic                 in_pp_wr_en;
    logic [AW_IN-1:0]     in_pp_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] in_pp_wr_data;

    logic                 in_pp_rd_en;
    logic [AW_IN-1:0]     in_pp_rd_addr;
    logic signed [ACT_W_LOCAL-1:0] in_pp_rd_data;

    logic                 out_pp_rd_bank_sel;
    logic                 out_pp_rd_en;
    logic [AW_OUT-1:0]    out_pp_rd_addr;
    logic signed [ACT_W_LOCAL-1:0] out_pp_rd_data;

    logic                 out_pp_wr_en;
    logic [AW_OUT-1:0]    out_pp_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] out_pp_wr_data;

    axi_input #(
        .ACT_W     (ACT_W_LOCAL),
        .INPUT_SIZE(INPUT_SZ),
        .AW        (AW_IN),
        .AXIS_DATA_W(AXIS_DATA_W),
        .AXIS_KEEP_W(AXIS_KEEP_W)
    ) u_axi_input (
        .clk                (clk),
        .rstn               (rstn),

        .s_axis_s2mm_tdata  (s_axis_s2mm_tdata),
        .s_axis_s2mm_tkeep  (s_axis_s2mm_tkeep),
        .s_axis_s2mm_tlast  (s_axis_s2mm_tlast),
        .s_axis_s2mm_tvalid (s_axis_s2mm_tvalid),
        .s_axis_s2mm_tready (s_axis_s2mm_tready),

        .phase_sel          (phase_sel),
        .phase_active       (phase_active),
        .consume_pulse      (in_consume_pulse),
        .bank_full          (in_bank_full),

        .wr_bank_sel        (in_pp_wr_bank_sel),
        .wr_en              (in_pp_wr_en),
        .wr_addr            (in_pp_wr_addr),
        .wr_data            (in_pp_wr_data)
    );

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

    axi_output #(
        .ACT_W      (ACT_W_LOCAL),
        .OUTPUT_SIZE(OUTPUT_SZ),
        .AW         (AW_OUT),
        .AXIS_DATA_W   (AXIS_DATA_W),
        .AXIS_KEEP_W   (AXIS_KEEP_W)
    ) u_axi_output (
        .clk                (clk),
        .rstn               (rstn),

        .phase_sel          (phase_sel),
        .phase_active       (phase_active),
        .produce_pulse      (out_produce_pulse),
        .bank_full          (out_bank_full),

        .rd_bank_sel        (out_pp_rd_bank_sel),
        .rd_en              (out_pp_rd_en),
        .rd_addr            (out_pp_rd_addr),
        .rd_data            (out_pp_rd_data),

        .m_axis_mm2s_tdata  (m_axis_mm2s_tdata),
        .m_axis_mm2s_tkeep  (m_axis_mm2s_tkeep),
        .m_axis_mm2s_tlast  (m_axis_mm2s_tlast),
        .m_axis_mm2s_tready (m_axis_mm2s_tready),
        .m_axis_mm2s_tvalid (m_axis_mm2s_tvalid)
    );

endmodule

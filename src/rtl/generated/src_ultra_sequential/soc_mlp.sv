`include "scales.vh"

// -----------------------------------------------------------------------------
// soc_mlp.sv
// Ultra96 sequential top-level.
//
// This top mirrors the Lattice sequential architecture, but replaces the UART
// front-end/back-end with AXI-Stream interfaces for the DMA path.
//
// External protocol:
//   - input frame  = exactly INPUT_SZ AXI words
//   - output frame = exactly OUTPUT_SZ AXI words
//   - AXI words are AXIS_DATA_W bits, 16 bits by default
//   - the MLP core uses ACT_W for every input/inter-layer/output vector
//
// The generated placeholders are intentionally the same as the Lattice flow:
//   `N6
//   66
// -----------------------------------------------------------------------------

module SOC_MLP (
    input  logic                       clk,
    input  logic                       rstn,

    // AXI-Stream input from DMA/MM2S-to-core direction.
    input  logic [15:0]     s_axis_s2mm_tdata,
    input  logic [1:0]     s_axis_s2mm_tkeep,
    input  logic                       s_axis_s2mm_tlast,
    input  logic                       s_axis_s2mm_tvalid,
    output logic                       s_axis_s2mm_tready,

    // AXI-Stream output from core-to-DMA/S2MM direction.
    output logic [15:0]     m_axis_mm2s_tdata,
    output logic [1:0]     m_axis_mm2s_tkeep,
    output logic                       m_axis_mm2s_tlast,
    input  logic                       m_axis_mm2s_tready,
    output logic                       m_axis_mm2s_tvalid
);

    // -------------------------------------------------------------------------
    // Generated/internal dimensions.
    // -------------------------------------------------------------------------
    localparam int AXIS_DATA_W 	  = 16;
    localparam int AXIS_KEEP_W    = (AXIS_DATA_W + 7) / 8;
    localparam int ACT_W_LOCAL    = `ACT_W;
    localparam int DSP_PW_LOCAL   = `DSP_PW;
    localparam int INPUT_SZ       = `N0;
    localparam int OUTPUT_SZ      = `N6;
    localparam int NUM_LINEAR_CFG = `NUM_LINEAR;
    localparam int MAX_NEURONS    = 66;
    localparam int VEC_AW         = (MAX_NEURONS > 1) ? $clog2(MAX_NEURONS) : 1;

    // Ping-pong RAM for the sequential MLP:
    // layer 0 writes B, layer 1 writes A, layer 2 writes B, ...
    // Therefore the final bank is 1 if NUM_LINEAR is odd, 0 if it is even.
    localparam logic FINAL_BANK_SEL = ((NUM_LINEAR_CFG % 2) == 1) ? 1'b1 : 1'b0;

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
    // RAM ports driven by axi_input.
    // -------------------------------------------------------------------------
    logic                      axi_in_wr_bank_sel;
    logic                      axi_in_wr_en;
    logic [VEC_AW-1:0]         axi_in_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] axi_in_wr_data;

    // -------------------------------------------------------------------------
    // RAM ports driven by the sequential MLP engine.
    // -------------------------------------------------------------------------
    logic                      mlp_vec_rd_en;
    logic                      mlp_vec_rd_bank_sel;
    logic [VEC_AW-1:0]         mlp_vec_rd_addr;

    logic                      mlp_vec_wr_en;
    logic                      mlp_vec_wr_bank_sel;
    logic [VEC_AW-1:0]         mlp_vec_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] mlp_vec_wr_data;

    // -------------------------------------------------------------------------
    // RAM ports driven by axi_output.
    // -------------------------------------------------------------------------
    logic                      axi_out_rd_bank_sel;
    logic                      axi_out_rd_en;
    logic [VEC_AW-1:0]         axi_out_rd_addr;

    // -------------------------------------------------------------------------
    // MEM_A / MEM_B arbitration.
    // axi_input writes MEM_A when the MLP/output are idle.
    // mlp_sequential reads/writes A/B while it is busy.
    // axi_output reads the final bank when the MLP is done.
    // -------------------------------------------------------------------------
    logic                      ram_rd_en;
    logic                      ram_rd_bank_sel;
    logic [VEC_AW-1:0]         ram_rd_addr;
    logic signed [ACT_W_LOCAL-1:0] ram_rd_data;

    logic                      ram_wr_en;
    logic                      ram_wr_bank_sel;
    logic [VEC_AW-1:0]         ram_wr_addr;
    logic signed [ACT_W_LOCAL-1:0] ram_wr_data;

    logic                      mem_a_rd_en;
    logic                      mem_b_rd_en;
    logic                      mem_a_wr_en;
    logic                      mem_b_wr_en;
    logic signed [ACT_W_LOCAL-1:0] mem_a_rd_data;
    logic signed [ACT_W_LOCAL-1:0] mem_b_rd_data;
    logic                      ram_rd_bank_sel_q;

    assign input_accept_enable = !mlp_busy && !output_busy;

    assign ram_rd_en       = mlp_busy ? mlp_vec_rd_en       : axi_out_rd_en;
    assign ram_rd_bank_sel = mlp_busy ? mlp_vec_rd_bank_sel : axi_out_rd_bank_sel;
    assign ram_rd_addr     = mlp_busy ? mlp_vec_rd_addr     : axi_out_rd_addr;

    assign ram_wr_en       = mlp_busy ? mlp_vec_wr_en       : axi_in_wr_en;
    assign ram_wr_bank_sel = mlp_busy ? mlp_vec_wr_bank_sel : axi_in_wr_bank_sel;
    assign ram_wr_addr     = mlp_busy ? mlp_vec_wr_addr     : axi_in_wr_addr;
    assign ram_wr_data     = mlp_busy ? mlp_vec_wr_data     : axi_in_wr_data;

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
    // AXI input: receives INPUT_SZ signed AXIS_DATA_W-bit words and loads MEM_A.
    // The module keeps the UART-like fixed-frame behavior: the frame closes
    // after INPUT_SIZE accepted words, not from TLAST.
    // -------------------------------------------------------------------------
    axi_input #(
        .ACT_W      (ACT_W_LOCAL),
        .INPUT_SIZE (INPUT_SZ),
        .AW         (VEC_AW),
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
        .accept_enable      (input_accept_enable),
        .consume_pulse      (input_consume_pulse),
        .frame_full         (input_frame_full),
        .loading            (input_loading),
        .wr_bank_sel        (axi_in_wr_bank_sel),
        .wr_en              (axi_in_wr_en),
        .wr_addr            (axi_in_wr_addr),
        .wr_data            (axi_in_wr_data)
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
    // AXI output: reads the final output vector from the MLP result bank and
    // sends OUTPUT_SZ signed AXIS_DATA_W-bit words with TLAST on the last word.
    // -------------------------------------------------------------------------
    axi_output #(
        .ACT_W         (ACT_W_LOCAL),
        .OUTPUT_SIZE   (OUTPUT_SZ),
        .AW            (VEC_AW),
        .FINAL_BANK_SEL(FINAL_BANK_SEL),
        .AXIS_DATA_W   (AXIS_DATA_W),
        .AXIS_KEEP_W   (AXIS_KEEP_W)
    ) u_axi_output (
        .clk                (clk),
        .rstn               (rstn),
        .produce_pulse      (output_produce_pulse),
        .busy               (output_busy),
        .done_pulse         (output_done_pulse),
        .rd_bank_sel        (axi_out_rd_bank_sel),
        .rd_en              (axi_out_rd_en),
        .rd_addr            (axi_out_rd_addr),
        .rd_data            (ram_rd_data),
        .m_axis_mm2s_tdata  (m_axis_mm2s_tdata),
        .m_axis_mm2s_tkeep  (m_axis_mm2s_tkeep),
        .m_axis_mm2s_tlast  (m_axis_mm2s_tlast),
        .m_axis_mm2s_tready (m_axis_mm2s_tready),
        .m_axis_mm2s_tvalid (m_axis_mm2s_tvalid)
    );

    wire _unused_ok = &{1'b0, input_loading, output_done_pulse, mlp_done_pulse};

endmodule

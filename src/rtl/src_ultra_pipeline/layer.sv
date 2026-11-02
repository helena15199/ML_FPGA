// -----------------------------------------------------------------------------
// layer.sv
// Single pipeline MLP layer.
//
// This module connects:
//   - the layer controller
//   - the layer datapath
//   - one weight ROM generated for this layer
//   - one bias ROM generated for this layer
//   - one optional activation LUT ROM generated for this layer
//   - one DSP/MAC block
//
// ROM contents are hard-coded in generated ROM modules. No external .mem file is
// loaded by this layer.
// -----------------------------------------------------------------------------

module {module_layer_name} #(
    parameter int ACT_TYPE          = 2,  // 0=identity, 1=relu, 2=leaky_relu_pow2, 3=lut_interp

    parameter int N_IN              = 12,
    parameter int N_OUT             = 80,

    parameter int ACT_W             = 16,
    parameter int WEIGHT_W          = 16,
    parameter int BIAS_W            = 16,

    parameter int LEAK_SHIFT        = 6,
    parameter bit HAS_SP            = 1'b1,
    parameter int Sx                = 15,
    parameter int Sw                = 13,
    parameter int Sb                = 18,
    parameter int Sp                = 16,
    parameter int Sx_out            = 14,
    parameter int LUT_N             = 0,

    // Accumulator width:
    //   - 32 bits for the Lattice/iCE40 flow
    //   - 48 bits for the Ultra96/Xilinx flow
    parameter int DSP_PW            = 32
)(
    input  logic clk,
    input  logic rstn,
    input  logic start,

    output logic                                       in_rd_en,
    output logic [((N_IN>1)?$clog2(N_IN):1)-1:0]       in_rd_addr,
    input  logic signed [ACT_W-1:0]                    in_rd_data,

    output logic                                       out_wr_en,
    output logic [((N_OUT>1)?$clog2(N_OUT):1)-1:0]     out_wr_addr,
    output logic signed [ACT_W-1:0]                    out_wr_data,

    output logic                                       done
);

    localparam logic USE_LUT_ACT = ((ACT_TYPE == 3) && (LUT_N > 0));
    localparam int LUTW          = (LUT_N > 0) ? LUT_N : 1;
    localparam int ACT_FRAC_W    = (ACT_W > LUTW) ? (ACT_W - LUTW) : 0;
    localparam int ACT_FRAC_EW   = (ACT_FRAC_W > 0) ? ACT_FRAC_W : 1;
    localparam int LUTN_W        = $clog2(ACT_W + 1);

    localparam int AW_W = ((N_IN*N_OUT) > 1) ? $clog2(N_IN*N_OUT) : 1;
    localparam int AW_B = (N_OUT > 1)       ? $clog2(N_OUT)       : 1;
    localparam int AW_A = (LUT_N > 0)       ? LUT_N              : 1;

    // The extra bit is required by yb-ya during LUT interpolation.
    localparam int DSP_AW = ACT_W + 1;
    localparam int DSP_BW = (WEIGHT_W > ACT_W) ? WEIGHT_W : ACT_W;

    logic [AW_W-1:0] w_addr;
    logic [AW_B-1:0] b_addr;

    logic       dsp_ce;
    logic       dsp_first_mac;
    logic [1:0] dsp_mode;

    logic signed [WEIGHT_W-1:0] weight_data;
    logic signed [BIAS_W-1:0]   bias_data;

    logic                         act_rom_en;

    logic signed [ACT_W-1:0]     act_x;
    logic signed [ACT_W-1:0]     act_ya;
    logic signed [ACT_W-1:0]     act_yb;
    logic [ACT_W-1:0]            act_frac_rom;
    logic [ACT_FRAC_EW-1:0]      act_frac;

    logic signed [DSP_AW-1:0]    dsp_A;
    logic signed [DSP_BW-1:0]    dsp_B;
    logic signed [DSP_PW-1:0]    dsp_C;
    logic signed [DSP_PW-1:0]    dsp_P;

    assign act_frac = act_frac_rom[ACT_FRAC_EW-1:0];

    // -------------------------------------------------------------------------
    // Control FSM: counters, addresses and sequencing enables.
    // -------------------------------------------------------------------------
    layer_ctrl #(
        .ACT_TYPE(ACT_TYPE),
        .N_IN    (N_IN),
        .N_OUT   (N_OUT),
        .LUT_N   (LUT_N)
    ) u_ctrl (
        .clk          (clk),
        .rstn         (rstn),
        .start        (start),

        .in_rd_en     (in_rd_en),
        .in_rd_addr   (in_rd_addr),

        .out_wr_en    (out_wr_en),
        .out_wr_addr  (out_wr_addr),

        .w_addr       (w_addr),
        .b_addr       (b_addr),

        .dsp_ce       (dsp_ce),
        .dsp_first_mac(dsp_first_mac),
        .dsp_mode     (dsp_mode),

        .act_rom_en   (act_rom_en),

        .done         (done)
    );

    // -------------------------------------------------------------------------
    // Layer-local ROMs.
    // Each generated layer instantiates its own generated ROM modules.
    // The ROM contents are initialized inside those modules by Python.
    // -------------------------------------------------------------------------
    {module_weight_name} #(
        .WEIGHT_W  (WEIGHT_W),
        .DEPTH     (N_IN * N_OUT),
        .ADDR_WIDTH(AW_W)
    ) u_w_rom (
        .clk (clk),
        .en  (1'b1),
        .addr(w_addr),
        .data(weight_data)
    );

    {module_bias_name} #(
        .BIAS_W    (BIAS_W),
        .DEPTH     (N_OUT),
        .ADDR_WIDTH(AW_B)
    ) u_b_rom (
        .clk (clk),
        .en  (1'b1),
        .addr(b_addr),
        .data(bias_data)
    );

    // -------------------------------------------------------------------------
    // Optional activation LUT ROM.
    // Non-LUT activations are handled directly inside the datapath.
    // -------------------------------------------------------------------------
    generate
        if (USE_LUT_ACT) begin : g_lut_act
            {module_act_name} #(
                .ACT_W     (ACT_W),
                .DEPTH     (1 << LUT_N),
                .ADDR_WIDTH(AW_A)
            ) u_rom_act (
                .clk  (clk),
                .en   (act_rom_en),

                .lut_n(LUTN_W'(LUT_N)),
                .x    (act_x),

                .ya   (act_ya),
                .yb   (act_yb),
                .frac (act_frac_rom)
            );
        end else begin : g_no_lut_act
            always_comb begin
                act_ya       = '0;
                act_yb       = '0;
                act_frac_rom = '0;
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Datapath: scaling, DSP input selection, direct activations and output.
    // ROMs and DSP are intentionally instantiated at this level.
    // -------------------------------------------------------------------------
    layer_datapath #(
        .ACT_TYPE        (ACT_TYPE),
        .ACT_W           (ACT_W),
        .WEIGHT_W        (WEIGHT_W),
        .BIAS_W          (BIAS_W),
        .LEAK_SHIFT      (LEAK_SHIFT),
        .HAS_SP          (HAS_SP),
        .Sx              (Sx),
        .Sw              (Sw),
        .Sb              (Sb),
        .Sp              (Sp),
        .Sx_out          (Sx_out),
        .LUT_N           (LUT_N),
        .DSP_AW          (DSP_AW),
        .ACT_FRAC_EW     (ACT_FRAC_EW),
        .DSP_BW          (DSP_BW),
        .DSP_PW          (DSP_PW)
    ) u_datapath (
        .in_data    (in_rd_data),
        .weight_data(weight_data),
        .bias_data  (bias_data),

        .dsp_P      (dsp_P),
        .dsp_mode   (dsp_mode),

        .act_ya     (act_ya),
        .act_yb     (act_yb),
        .act_frac   (act_frac),
        .act_x      (act_x),

        .dsp_A      (dsp_A),
        .dsp_B      (dsp_B),
        .dsp_C      (dsp_C),

        .out_data   (out_wr_data)
    );

    // -------------------------------------------------------------------------
    // DSP/MAC block.
    // -----------------------------------------------------------------------------
    dsp #(
        .A_WIDTH(DSP_AW),
        .B_WIDTH(DSP_BW),
        .P_WIDTH(DSP_PW)
    ) u_dsp (
        .clk      (clk),
        .rstn     (rstn),
        .ce       (dsp_ce),
        .first_mac(dsp_first_mac),
        .A        (dsp_A),
        .B        (dsp_B),
        .C        (dsp_C),
        .P        (dsp_P)
    );

endmodule

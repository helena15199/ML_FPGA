// -----------------------------------------------------------------------------
// mlp_sequential_datapath.sv - V2 timing-optimized
// Shared datapath for the sequential MLP engine.
//
// Scale arithmetic is elaboration-time only. Each generated layer branch uses
// constant shifts, so the synthesized design contains wiring/muxing rather than
// variable barrel shifters driven by run-time scale calculations.
// -----------------------------------------------------------------------------

`include "scales.vh"

module mlp_sequential_datapath #(
    parameter int NUM_LINEAR_CFG = `NUM_LINEAR,
    parameter int ACT_W          = `ACT_W,
    parameter int WEIGHT_W       = `WEIGHT_W,
    parameter int BIAS_W         = `BIAS_W,
    parameter int DSP_PW         = `DSP_PW,
    parameter int LEAK_SHIFT     = `LEAK_SHIFT,
    parameter int ACT_ADDR_W     = 8,
    parameter int LAYER_AW       = (NUM_LINEAR_CFG > 1) ? $clog2(NUM_LINEAR_CFG) : 1,
    parameter int DSP_AW         = ACT_W + 1,
    parameter int DSP_BW         = ((WEIGHT_W > ACT_W) ? WEIGHT_W : ACT_W)
)(
    input  logic                         clk,
    input  logic                         rstn,

    input  logic [LAYER_AW-1:0]          layer_idx,
    input  logic                         load_layer_cfg,

    input  logic signed [ACT_W-1:0]      vec_rd_data,
    input  logic signed [WEIGHT_W-1:0]   weight_data,
    input  logic signed [BIAS_W-1:0]     bias_data,

    input  logic signed [DSP_PW-1:0]     dsp_P,
    input  logic [1:0]                   dsp_mode,

    input  logic signed [ACT_W-1:0]      act_ya,
    input  logic signed [ACT_W-1:0]      act_yb,
    input  logic [ACT_W-1:0]             act_frac,

    output logic                         use_lut_act,
    output logic signed [ACT_W-1:0]      act_x,
    output logic [ACT_ADDR_W-1:0]        act_base_addr,
    output logic [$clog2(ACT_W+1)-1:0]   act_lut_n,

    output logic signed [DSP_AW-1:0]     dsp_A,
    output logic signed [DSP_BW-1:0]     dsp_B,
    output logic signed [DSP_PW-1:0]     dsp_C,

    output logic signed [ACT_W-1:0]      out_data
);

    localparam logic [1:0] DSP_MODE_IDLE       = 2'd0;
    localparam logic [1:0] DSP_MODE_MAC        = 2'd1;
    localparam logic [1:0] DSP_MODE_ACT_INTERP = 2'd2;
    localparam int LUTN_W = $clog2(ACT_W+1);

    localparam int ACT_DEPTH0 = ((`ACT0 == 3) && (`LUT_N0 > 0)) ? (1 << `LUT_N0) : 0;
    localparam int ACT_DEPTH1 = ((`ACT1 == 3) && (`LUT_N1 > 0)) ? (1 << `LUT_N1) : 0;
    localparam int ACT_DEPTH2 = ((`ACT2 == 3) && (`LUT_N2 > 0)) ? (1 << `LUT_N2) : 0;
    localparam int ACT_DEPTH3 = ((`ACT3 == 3) && (`LUT_N3 > 0)) ? (1 << `LUT_N3) : 0;
    localparam int ACT_DEPTH4 = ((`ACT4 == 3) && (`LUT_N4 > 0)) ? (1 << `LUT_N4) : 0;
    localparam int ACT_DEPTH5 = ((`ACT5 == 3) && (`LUT_N5 > 0)) ? (1 << `LUT_N5) : 0;
    localparam int ACT_DEPTH_RAW = ACT_DEPTH0 + ACT_DEPTH1 + ACT_DEPTH2 + ACT_DEPTH3 + ACT_DEPTH4 + ACT_DEPTH5;
    localparam int ACT_DEPTH = (ACT_DEPTH_RAW > 0) ? ACT_DEPTH_RAW : 1;

    function automatic int sx_at(input int l);
        begin
            unique case (l)
                0: sx_at = `SX0;
                1: sx_at = `SX1;
                2: sx_at = `SX2;
                3: sx_at = `SX3;
                4: sx_at = `SX4;
                5: sx_at = `SX5;
                6: sx_at = `SX6;
                default: sx_at = `SX0;
            endcase
        end
    endfunction

    function automatic int sw_at(input int l);
        begin
            unique case (l)
                0: sw_at = `SW0;
                1: sw_at = `SW1;
                2: sw_at = `SW2;
                3: sw_at = `SW3;
                4: sw_at = `SW4;
                5: sw_at = `SW5;
                default: sw_at = `SW0;
            endcase
        end
    endfunction

    function automatic int sb_at(input int l);
        begin
            unique case (l)
                0: sb_at = `SB0;
                1: sb_at = `SB1;
                2: sb_at = `SB2;
                3: sb_at = `SB3;
                4: sb_at = `SB4;
                5: sb_at = `SB5;
                default: sb_at = `SB0;
            endcase
        end
    endfunction

    function automatic int sp_at(input int l);
        begin
            unique case (l)
                0: sp_at = `SP0;
                1: sp_at = `SP1;
                2: sp_at = `SP2;
                3: sp_at = `SP3;
                4: sp_at = `SP4;
                default: sp_at = sx_at(l + 1);
            endcase
        end
    endfunction

    function automatic int act_at(input int l);
        begin
            unique case (l)
                0: act_at = `ACT0;
                1: act_at = `ACT1;
                2: act_at = `ACT2;
                3: act_at = `ACT3;
                4: act_at = `ACT4;
                5: act_at = `ACT5;
                default: act_at = 0;
            endcase
        end
    endfunction

    function automatic int lutn_at(input int l);
        begin
            unique case (l)
                0: lutn_at = `LUT_N0;
                1: lutn_at = `LUT_N1;
                2: lutn_at = `LUT_N2;
                3: lutn_at = `LUT_N3;
                4: lutn_at = `LUT_N4;
                5: lutn_at = `LUT_N5;
                default: lutn_at = 0;
            endcase
        end
    endfunction

    function automatic int act_base_at(input int l);
        begin
            unique case (l)
                0: act_base_at = 0;
                1: act_base_at = ACT_DEPTH0;
                2: act_base_at = ACT_DEPTH0 + ACT_DEPTH1;
                3: act_base_at = ACT_DEPTH0 + ACT_DEPTH1 + ACT_DEPTH2;
                4: act_base_at = ACT_DEPTH0 + ACT_DEPTH1 + ACT_DEPTH2 + ACT_DEPTH3;
                5: act_base_at = ACT_DEPTH0 + ACT_DEPTH1 + ACT_DEPTH2 + ACT_DEPTH3 + ACT_DEPTH4;
                default: act_base_at = 0;
            endcase
        end
    endfunction

    function automatic logic signed [DSP_PW-1:0]
        shift_const_p(input logic signed [DSP_PW-1:0] v, input integer sh);
        begin
            if (sh == 0)      shift_const_p = v;
            else if (sh > 0)  shift_const_p = (v <<< sh);
            else              shift_const_p = (v >>> (-sh));
        end
    endfunction

    function automatic logic signed [ACT_W-1:0]
        to_act_width(input logic signed [DSP_PW-1:0] v);
        begin
            to_act_width = v[ACT_W-1:0];
        end
    endfunction

    function automatic logic signed [DSP_PW-1:0]
        sign_extend_act_to_p(input logic signed [ACT_W-1:0] v);
        begin
            sign_extend_act_to_p = $signed(v);
        end
    endfunction

    logic [LAYER_AW-1:0] layer_cfg_r;
    integer act_r;
    integer lut_n_r;
    integer act_base_r;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            layer_cfg_r <= '0;
            act_r       <= 0;
            lut_n_r     <= 0;
            act_base_r  <= 0;
        end else if (load_layer_cfg) begin
            layer_cfg_r <= layer_idx;
            act_r       <= act_at($unsigned(layer_idx));
            lut_n_r     <= lutn_at($unsigned(layer_idx));
            act_base_r  <= act_base_at($unsigned(layer_idx));
        end
    end

    logic signed [DSP_PW-1:0] bias_p;
    logic signed [DSP_PW-1:0] simple_act_p;

    logic signed [DSP_PW-1:0] bias_aligned_by_layer [0:NUM_LINEAR_CFG-1];
    logic signed [DSP_PW-1:0] pre_aligned_by_layer  [0:NUM_LINEAR_CFG-1];
    logic signed [DSP_PW-1:0] out_aligned_by_layer  [0:NUM_LINEAR_CFG-1];
    logic signed [DSP_PW-1:0] act_c_by_layer        [0:NUM_LINEAR_CFG-1];
    logic signed [DSP_PW-1:0] interp_by_layer       [0:NUM_LINEAR_CFG-1];

    assign bias_p = $signed(bias_data);

    always_comb begin
        unique case (act_r)
            0: simple_act_p = dsp_P;
            1: simple_act_p = dsp_P[DSP_PW-1] ? '0 : dsp_P;
            2: simple_act_p = dsp_P[DSP_PW-1] ? (dsp_P >>> LEAK_SHIFT) : dsp_P;
            default: simple_act_p = dsp_P;
        endcase
    end

    generate
        for (genvar gi = 0; gi < NUM_LINEAR_CFG; gi = gi + 1) begin : g_const_scale
            localparam int SACC_G = sx_at(gi) + sw_at(gi);
            localparam int BIAS_SHIFT_G = SACC_G - sb_at(gi);
            localparam int PRE_SHIFT_G = (gi < (NUM_LINEAR_CFG-1)) ? (sp_at(gi) - SACC_G) : 0;
            localparam int OUT_SHIFT_G = sx_at(gi + 1) - SACC_G;
            localparam int FRAC_SHIFT_G =
                ((act_at(gi) == 3) && (lutn_at(gi) > 0) && (ACT_W > lutn_at(gi)))
                ? (ACT_W - lutn_at(gi)) : 0;

            assign bias_aligned_by_layer[gi] = shift_const_p(bias_p, BIAS_SHIFT_G);
            assign pre_aligned_by_layer[gi]  = shift_const_p(dsp_P, PRE_SHIFT_G);
            assign out_aligned_by_layer[gi]  = shift_const_p(simple_act_p, OUT_SHIFT_G);
            assign act_c_by_layer[gi]        = shift_const_p(sign_extend_act_to_p(act_ya), FRAC_SHIFT_G);
            assign interp_by_layer[gi]       = shift_const_p(dsp_P, -FRAC_SHIFT_G);
        end
    endgenerate

    logic signed [DSP_PW-1:0] C_bias_p;
    logic signed [DSP_PW-1:0] pre_p_w;

    logic signed [DSP_AW-1:0] mac_A_w;
    logic signed [DSP_AW-1:0] act_A_w;
    logic signed [DSP_AW-1:0] act_yb_ext_w;
    logic signed [DSP_AW-1:0] act_ya_ext_w;

    logic signed [DSP_BW-1:0] mac_B_w;
    logic signed [DSP_BW-1:0] act_B_w;
    logic [DSP_BW-1:0]        act_frac_b_w;

    logic signed [DSP_PW-1:0] act_C_w;

    assign C_bias_p = bias_aligned_by_layer[layer_cfg_r];
    assign pre_p_w  = pre_aligned_by_layer[layer_cfg_r];
    assign act_x    = to_act_width(pre_p_w);

    assign use_lut_act   = ((act_r == 3) && (lut_n_r > 0));
    assign act_base_addr = ACT_ADDR_W'(act_base_r);
    assign act_lut_n     = LUTN_W'(lut_n_r);

    assign mac_A_w      = $signed(vec_rd_data);
    assign mac_B_w      = $signed(weight_data);

    assign act_yb_ext_w = $signed(act_yb);
    assign act_ya_ext_w = $signed(act_ya);
    assign act_A_w      = act_yb_ext_w - act_ya_ext_w;

    assign act_frac_b_w = DSP_BW'($unsigned(act_frac));
    assign act_B_w      = $signed(act_frac_b_w);
    assign act_C_w      = act_c_by_layer[layer_cfg_r];

    always_comb begin
        dsp_A = '0;
        dsp_B = '0;
        dsp_C = C_bias_p;

        unique case (dsp_mode)
            DSP_MODE_MAC: begin
                dsp_A = mac_A_w;
                dsp_B = mac_B_w;
                dsp_C = C_bias_p;
            end

            DSP_MODE_ACT_INTERP: begin
                dsp_A = act_A_w;
                dsp_B = act_B_w;
                dsp_C = act_C_w;
            end

            default: begin
                dsp_A = '0;
                dsp_B = '0;
                dsp_C = C_bias_p;
            end
        endcase
    end

    always_comb begin
        if (use_lut_act)
            out_data = interp_by_layer[layer_cfg_r][ACT_W-1:0];
        else
            out_data = out_aligned_by_layer[layer_cfg_r][ACT_W-1:0];
    end

endmodule
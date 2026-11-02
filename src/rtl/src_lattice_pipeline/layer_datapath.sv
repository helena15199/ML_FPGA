// -----------------------------------------------------------------------------
// layer_datapath.sv
// Datapath of one MLP layer. All scale-derived shifts are localparams and are
// therefore compile-time constants for each generated layer instance.
// -----------------------------------------------------------------------------

module layer_datapath #(
    parameter int ACT_TYPE          = 2,

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

    parameter int DSP_PW            = 32,
    parameter int DSP_AW            = ACT_W + 1,
    parameter int ACT_FRAC_EW       = 1,
    parameter int DSP_BW            = (WEIGHT_W > ACT_W) ? WEIGHT_W : ACT_W
)(
    input  logic signed [ACT_W-1:0]             in_data,
    input  logic signed [WEIGHT_W-1:0]          weight_data,
    input  logic signed [BIAS_W-1:0]            bias_data,

    input  logic signed [DSP_PW-1:0]            dsp_P,
    input  logic [1:0]                          dsp_mode,

    input  logic signed [ACT_W-1:0]             act_ya,
    input  logic signed [ACT_W-1:0]             act_yb,
    input  logic [ACT_FRAC_EW-1:0]              act_frac,

    output logic signed [ACT_W-1:0]             act_x,

    output logic signed [DSP_AW-1:0]            dsp_A,
    output logic signed [DSP_BW-1:0]            dsp_B,
    output logic signed [DSP_PW-1:0]            dsp_C,

    output logic signed [ACT_W-1:0]             out_data
);

    localparam int PW = DSP_PW;

    localparam bit USE_LUT_ACT = ((ACT_TYPE == 3) && (LUT_N > 0));
    localparam int LUTW        = (LUT_N > 0) ? LUT_N : 1;
    localparam int ACT_FRAC_W  = (ACT_W > LUTW) ? (ACT_W - LUTW) : 0;

    localparam int S_ACC      = Sx + Sw;
    localparam int BIAS_SHIFT = S_ACC - Sb;
    localparam int PRE_SHIFT  = HAS_SP ? (Sp - S_ACC) : 0;
    localparam int OUT_SHIFT  = Sx_out - S_ACC;

    localparam logic [1:0] DSP_MODE_IDLE       = 2'd0;
    localparam logic [1:0] DSP_MODE_MAC        = 2'd1;
    localparam logic [1:0] DSP_MODE_ACT_INTERP = 2'd2;

    function automatic logic signed [PW-1:0]
        align_bias(input logic signed [PW-1:0] v);
        begin
            if (BIAS_SHIFT == 0)      align_bias = v;
            else if (BIAS_SHIFT > 0)  align_bias = v <<< BIAS_SHIFT;
            else                      align_bias = v >>> (-BIAS_SHIFT);
        end
    endfunction

    function automatic logic signed [PW-1:0]
        align_preact(input logic signed [PW-1:0] v);
        begin
            if (PRE_SHIFT == 0)      align_preact = v;
            else if (PRE_SHIFT > 0)  align_preact = v <<< PRE_SHIFT;
            else                     align_preact = v >>> (-PRE_SHIFT);
        end
    endfunction

    function automatic logic signed [PW-1:0]
        align_output(input logic signed [PW-1:0] v);
        begin
            if (OUT_SHIFT == 0)      align_output = v;
            else if (OUT_SHIFT > 0)  align_output = v <<< OUT_SHIFT;
            else                     align_output = v >>> (-OUT_SHIFT);
        end
    endfunction

    function automatic logic signed [ACT_W-1:0]
        to_act_width(input logic signed [PW-1:0] v);
        begin
            to_act_width = v[ACT_W-1:0];
        end
    endfunction

    function automatic logic signed [PW-1:0]
        sign_extend_to_pw(input logic signed [ACT_W-1:0] v);
        begin
            sign_extend_to_pw = {{(PW-ACT_W){v[ACT_W-1]}}, v};
        end
    endfunction

    function automatic logic signed [PW-1:0]
        make_act_c(input logic signed [ACT_W-1:0] ya);
        logic signed [PW-1:0] ya_pw;
        begin
            ya_pw = sign_extend_to_pw(ya);
            if (ACT_FRAC_W > 0) make_act_c = ya_pw <<< ACT_FRAC_W;
            else                make_act_c = ya_pw;
        end
    endfunction

    function automatic logic signed [ACT_W-1:0]
        quantize_simple(input logic signed [PW-1:0] acc_p);
        logic signed [PW-1:0] act_p;
        logic signed [PW-1:0] out_p;
        begin
            unique case (ACT_TYPE)
                0: act_p = acc_p;
                1: act_p = acc_p[PW-1] ? '0 : acc_p;
                2: act_p = acc_p[PW-1] ? (acc_p >>> LEAK_SHIFT) : acc_p;
                default: act_p = acc_p;
            endcase

            out_p = align_output(act_p);
            quantize_simple = out_p[ACT_W-1:0];
        end
    endfunction

    function automatic logic signed [ACT_W-1:0]
        quantize_lut_interp(input logic signed [PW-1:0] interp);
        logic signed [PW-1:0] out;
        begin
            if (ACT_FRAC_W > 0) out = interp >>> ACT_FRAC_W;
            else                out = interp;
            quantize_lut_interp = out[ACT_W-1:0];
        end
    endfunction

    logic signed [PW-1:0] bias_pw;
    logic signed [PW-1:0] C_bias;
    logic signed [PW-1:0] pre_w;

    logic signed [DSP_AW-1:0] mac_A_w;
    logic signed [DSP_AW-1:0] act_A_w;
    logic signed [DSP_BW-1:0] mac_B_w;
    logic signed [DSP_BW-1:0] act_B_w;
    logic signed [PW-1:0]     act_C_w;

    assign bias_pw = {{(PW-BIAS_W){bias_data[BIAS_W-1]}}, bias_data};
    assign C_bias  = align_bias(bias_pw);

    assign pre_w = align_preact(dsp_P);
    assign act_x = to_act_width(pre_w);

    assign mac_A_w = {{(DSP_AW-ACT_W){in_data[ACT_W-1]}}, in_data};
    assign mac_B_w = {{(DSP_BW-WEIGHT_W){weight_data[WEIGHT_W-1]}}, weight_data};

    assign act_A_w =
        {{(DSP_AW-ACT_W){act_yb[ACT_W-1]}}, act_yb} -
        {{(DSP_AW-ACT_W){act_ya[ACT_W-1]}}, act_ya};

    assign act_B_w = {{(DSP_BW-ACT_FRAC_EW){1'b0}}, act_frac};
    assign act_C_w = make_act_c(act_ya);

    always_comb begin
        dsp_A = '0;
        dsp_B = '0;
        dsp_C = C_bias;

        unique case (dsp_mode)
            DSP_MODE_MAC: begin
                dsp_A = mac_A_w;
                dsp_B = mac_B_w;
                dsp_C = C_bias;
            end

            DSP_MODE_ACT_INTERP: begin
                dsp_A = act_A_w;
                dsp_B = act_B_w;
                dsp_C = act_C_w;
            end

            default: begin
                dsp_A = '0;
                dsp_B = '0;
                dsp_C = C_bias;
            end
        endcase
    end

    always_comb begin
        out_data = USE_LUT_ACT ? quantize_lut_interp(dsp_P) : quantize_simple(dsp_P);
    end

endmodule
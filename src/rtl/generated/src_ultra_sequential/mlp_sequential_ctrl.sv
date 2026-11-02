`include "scales.vh"

module mlp_sequential_ctrl #(
    parameter int INPUT_SIZE     = `N0,
    parameter int OUTPUT_SIZE    = `N6,
    parameter int NUM_LINEAR_CFG = `NUM_LINEAR,

    parameter int VEC_AW         = 6,
    parameter int WEIGHT_ADDR_W  = 12,
    parameter int BIAS_ADDR_W    = 8,
    parameter int LAYER_AW       = (NUM_LINEAR_CFG > 1) ? $clog2(NUM_LINEAR_CFG) : 1
)(
    input  logic                         clk,
    input  logic                         rstn,

    input  logic                         input_frame_full,
    input  logic                         output_busy,
    output logic                         input_consume_pulse,
    output logic                         output_produce_pulse,

    output logic [LAYER_AW-1:0]          layer_idx,
    output logic                         load_layer_cfg,
    input  logic                         use_lut_act,

    output logic                         vec_rd_en,
    output logic                         vec_rd_bank_sel,
    output logic [VEC_AW-1:0]            vec_rd_addr,

    output logic                         vec_wr_en,
    output logic                         vec_wr_bank_sel,
    output logic [VEC_AW-1:0]            vec_wr_addr,

    output logic                         weight_en,
    output logic [WEIGHT_ADDR_W-1:0]     weight_addr,
    output logic                         bias_en,
    output logic [BIAS_ADDR_W-1:0]       bias_addr,

    output logic                         act_rom_en,

    output logic                         dsp_ce,
    output logic                         dsp_first_mac,
    output logic [1:0]                   dsp_mode,

    output logic                         busy,
    output logic                         compute_done_pulse
);

    localparam logic [1:0] DSP_MODE_IDLE       = 2'd0;
    localparam logic [1:0] DSP_MODE_MAC        = 2'd1;
    localparam logic [1:0] DSP_MODE_ACT_INTERP = 2'd2;

    typedef enum logic [3:0] {
        S_IDLE,
        S_LAYER_INIT,
        S_NEURON_PREP,
        S_MAC,
        S_ACT_RD,
        S_ACT_DSP,
        S_WRITE
    } state_t;

    state_t state, state_n;

    logic [LAYER_AW-1:0] layer_idx_n;
    logic [VEC_AW-1:0]   neuron_idx, neuron_idx_n;
    logic [VEC_AW-1:0]   i_idx, i_idx_n;

    logic [WEIGHT_ADDR_W-1:0] weight_ptr, weight_ptr_n;
    logic [BIAS_ADDR_W-1:0]   bias_ptr, bias_ptr_n;

    logic [VEC_AW-1:0] n_in_r,  n_in_r_n;
    logic [VEC_AW-1:0] n_out_r, n_out_r_n;

    function automatic int n_at(input int l);
        begin
            unique case (l)
                0: n_at = `N0;
                1: n_at = `N1;
                2: n_at = `N2;
                3: n_at = `N3;
                4: n_at = `N4;
                5: n_at = `N5;
                6: n_at = `N6;
                default: n_at = `N0;
            endcase
        end
    endfunction

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state      <= S_IDLE;
            layer_idx  <= '0;
            neuron_idx <= '0;
            i_idx      <= '0;
            weight_ptr <= '0;
            bias_ptr   <= '0;
            n_in_r     <= '0;
            n_out_r    <= '0;
        end else begin
            state      <= state_n;
            layer_idx  <= layer_idx_n;
            neuron_idx <= neuron_idx_n;
            i_idx      <= i_idx_n;
            weight_ptr <= weight_ptr_n;
            bias_ptr   <= bias_ptr_n;
            n_in_r     <= n_in_r_n;
            n_out_r    <= n_out_r_n;
        end
    end

    always_comb begin
        state_n      = state;
        layer_idx_n  = layer_idx;
        neuron_idx_n = neuron_idx;
        i_idx_n      = i_idx;
        weight_ptr_n = weight_ptr;
        bias_ptr_n   = bias_ptr;
        n_in_r_n     = n_in_r;
        n_out_r_n    = n_out_r;

        input_consume_pulse  = 1'b0;
        output_produce_pulse = 1'b0;
        compute_done_pulse   = 1'b0;

        load_layer_cfg = 1'b0;

        vec_rd_en       = 1'b0;
        vec_rd_bank_sel = layer_idx[0];
        vec_rd_addr     = i_idx;

        vec_wr_en       = 1'b0;
        vec_wr_bank_sel = ~layer_idx[0];
        vec_wr_addr     = neuron_idx;

        weight_en   = 1'b0;
        weight_addr = weight_ptr;
        bias_en     = 1'b0;
        bias_addr   = bias_ptr;

        act_rom_en = 1'b0;

        dsp_ce        = 1'b0;
        dsp_first_mac = 1'b0;
        dsp_mode      = DSP_MODE_IDLE;

        busy = (state != S_IDLE);

        unique case (state)
            S_IDLE: begin
                layer_idx_n  = '0;
                neuron_idx_n = '0;
                i_idx_n      = '0;
                weight_ptr_n = '0;
                bias_ptr_n   = '0;
                n_in_r_n     = '0;
                n_out_r_n    = '0;

                if (input_frame_full && !output_busy) begin
                    input_consume_pulse = 1'b1;
                    state_n             = S_LAYER_INIT;
                end
            end

            S_LAYER_INIT: begin
                load_layer_cfg = 1'b1;
                neuron_idx_n   = '0;
                i_idx_n        = '0;
                n_in_r_n       = VEC_AW'(n_at($unsigned(layer_idx)));
                n_out_r_n      = VEC_AW'(n_at($unsigned(layer_idx) + 1));
                state_n        = S_NEURON_PREP;
            end

            S_NEURON_PREP: begin
                i_idx_n = '0;

                bias_en    = 1'b1;
                bias_addr  = bias_ptr;
                bias_ptr_n = bias_ptr + BIAS_ADDR_W'(1);

                weight_en    = 1'b1;
                weight_addr  = weight_ptr;
                weight_ptr_n = weight_ptr + WEIGHT_ADDR_W'(1);

                vec_rd_en       = 1'b1;
                vec_rd_bank_sel = layer_idx[0];
                vec_rd_addr     = '0;

                state_n = S_MAC;
            end

            S_MAC: begin
                dsp_ce        = 1'b1;
                dsp_first_mac = ($unsigned(i_idx) == 0);
                dsp_mode      = DSP_MODE_MAC;

                if ($unsigned(i_idx) == ($unsigned(n_in_r) - 1)) begin
                    state_n = use_lut_act ? S_ACT_RD : S_WRITE;
                end else begin
                    weight_en    = 1'b1;
                    weight_addr  = weight_ptr;
                    weight_ptr_n = weight_ptr + WEIGHT_ADDR_W'(1);

                    vec_rd_en       = 1'b1;
                    vec_rd_bank_sel = layer_idx[0];
                    vec_rd_addr     = i_idx + VEC_AW'(1);

                    i_idx_n = i_idx + VEC_AW'(1);
                    state_n = S_MAC;
                end
            end

            S_ACT_RD: begin
                act_rom_en = 1'b1;
                state_n    = S_ACT_DSP;
            end

            S_ACT_DSP: begin
                dsp_ce        = 1'b1;
                dsp_first_mac = 1'b1;
                dsp_mode      = DSP_MODE_ACT_INTERP;
                state_n       = S_WRITE;
            end

            S_WRITE: begin
                vec_wr_en       = 1'b1;
                vec_wr_bank_sel = ~layer_idx[0];
                vec_wr_addr     = neuron_idx;

                if ($unsigned(neuron_idx) == ($unsigned(n_out_r) - 1)) begin
                    if ($unsigned(layer_idx) == (NUM_LINEAR_CFG - 1)) begin
                        output_produce_pulse = 1'b1;
                        compute_done_pulse   = 1'b1;
                        state_n              = S_IDLE;
                    end else begin
                        layer_idx_n  = layer_idx + LAYER_AW'(1);
                        neuron_idx_n = '0;
                        i_idx_n      = '0;
                        state_n      = S_LAYER_INIT;
                    end
                end else begin
                    neuron_idx_n = neuron_idx + VEC_AW'(1);
                    i_idx_n      = '0;
                    state_n      = S_NEURON_PREP;
                end
            end

            default: begin
                state_n = S_IDLE;
            end
        endcase
    end

endmodule
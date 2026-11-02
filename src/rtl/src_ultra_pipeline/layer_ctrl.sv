// -----------------------------------------------------------------------------
// layer_ctrl.sv
// Control FSM for a single MLP layer.
// Weight addresses use a sequential pointer because each generated layer ROM is
// already stored in neuron -> input execution order.
// -----------------------------------------------------------------------------

module layer_ctrl #(
    parameter int ACT_TYPE = 2,
    parameter int N_IN     = 12,
    parameter int N_OUT    = 80,
    parameter int LUT_N    = 0
)(
    input  logic clk,
    input  logic rstn,
    input  logic start,

    output logic                                       in_rd_en,
    output logic [((N_IN>1)?$clog2(N_IN):1)-1:0]       in_rd_addr,

    output logic                                       out_wr_en,
    output logic [((N_OUT>1)?$clog2(N_OUT):1)-1:0]     out_wr_addr,

    output logic [(((N_IN*N_OUT)>1)?$clog2(N_IN*N_OUT):1)-1:0] w_addr,
    output logic [((N_OUT>1)?$clog2(N_OUT):1)-1:0]             b_addr,

    output logic                                       dsp_ce,
    output logic                                       dsp_first_mac,
    output logic [1:0]                                 dsp_mode,

    output logic                                       act_rom_en,
    output logic                                       done
);

    localparam logic USE_LUT_ACT = ((ACT_TYPE == 3) && (LUT_N > 0));

    localparam int WI   = (N_IN  > 1) ? $clog2(N_IN ) : 1;
    localparam int WO   = (N_OUT > 1) ? $clog2(N_OUT) : 1;
    localparam int AW_W = ((N_IN*N_OUT) > 1) ? $clog2(N_IN*N_OUT) : 1;

    localparam logic [WI-1:0] LAST_IN  = WI'(N_IN  - 1);
    localparam logic [WO-1:0] LAST_OUT = WO'(N_OUT - 1);

    localparam logic [1:0] DSP_MODE_IDLE       = 2'd0;
    localparam logic [1:0] DSP_MODE_MAC        = 2'd1;
    localparam logic [1:0] DSP_MODE_ACT_INTERP = 2'd2;

    typedef enum logic [2:0] {
        S_IDLE,
        S_PREP,
        S_MAC,
        S_ACT_RD,
        S_ACT_DSP,
        S_WRITE,
        S_DONE
    } state_t;

    state_t state, state_n;

    logic [WO-1:0]   neuron, neuron_n;
    logic [WI-1:0]   i_idx, i_idx_n;
    logic [AW_W-1:0] weight_ptr, weight_ptr_n;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state      <= S_IDLE;
            neuron     <= '0;
            i_idx      <= '0;
            weight_ptr <= '0;
        end else begin
            state      <= state_n;
            neuron     <= neuron_n;
            i_idx      <= i_idx_n;
            weight_ptr <= weight_ptr_n;
        end
    end

    always_comb begin
        state_n      = state;
        neuron_n     = neuron;
        i_idx_n      = i_idx;
        weight_ptr_n = weight_ptr;

        in_rd_en    = 1'b0;
        in_rd_addr  = i_idx;

        out_wr_en   = 1'b0;
        out_wr_addr = neuron;

        b_addr = neuron;
        w_addr = weight_ptr;

        dsp_ce        = 1'b0;
        dsp_first_mac = 1'b0;
        dsp_mode      = DSP_MODE_IDLE;

        act_rom_en = 1'b0;
        done       = 1'b0;

        unique case (state)
            S_IDLE: begin
                neuron_n     = '0;
                i_idx_n      = '0;
                weight_ptr_n = '0;
                if (start) begin
                    state_n = S_PREP;
                end
            end

            S_PREP: begin
                b_addr       = neuron;
                w_addr       = weight_ptr;
                weight_ptr_n = weight_ptr + AW_W'(1);
                in_rd_en     = 1'b1;
                in_rd_addr   = '0;
                i_idx_n      = '0;
                state_n      = S_MAC;
            end

            S_MAC: begin
                dsp_ce        = 1'b1;
                dsp_first_mac = (i_idx == '0);
                dsp_mode      = DSP_MODE_MAC;

                if (i_idx == LAST_IN) begin
                    state_n = USE_LUT_ACT ? S_ACT_RD : S_WRITE;
                end else begin
                    in_rd_en     = 1'b1;
                    in_rd_addr   = i_idx + WI'(1);
                    w_addr       = weight_ptr;
                    weight_ptr_n = weight_ptr + AW_W'(1);
                    i_idx_n      = i_idx + WI'(1);
                    state_n      = S_MAC;
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
                out_wr_en   = 1'b1;
                out_wr_addr = neuron;

                if (neuron == LAST_OUT) begin
                    state_n = S_DONE;
                end else begin
                    neuron_n = neuron + WO'(1);
                    i_idx_n   = '0;
                    state_n   = S_PREP;
                end
            end

            S_DONE: begin
                done    = 1'b1;
                state_n = S_IDLE;
            end

            default: begin
                state_n = S_IDLE;
            end
        endcase
    end

endmodule
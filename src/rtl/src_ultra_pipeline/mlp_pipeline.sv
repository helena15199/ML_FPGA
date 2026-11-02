`include "scales.vh"

// -----------------------------------------------------------------------------
// mlp_pipeline.sv
// Pipeline MLP core.
//
// This module connects:
//   - the global phase controller
//   - the intermediate ping-pong buffers
//   - all generated layer modules
//
// The numeric values remain defined in scales.vh.
// Python only generates the structural parts that depend on the model topology:
//   - localparam tables
//   - intermediate buffers
//   - generated layer instances
//
// No external .mem file is loaded here. Each generated layer instantiates
// generated ROM modules whose contents are hard-coded by Python.
// -----------------------------------------------------------------------------

module mlp_pipeline #(
    parameter int ACT_W       = `ACT_W,
    parameter int WEIGHT_W    = `WEIGHT_W,
    parameter int BIAS_W      = `BIAS_W,
    parameter int DSP_PW      = `DSP_PW,
    parameter int INPUT_SIZE  = `N0,
    parameter int OUTPUT_SIZE = 2
)(
    input  logic                                                clk,
    input  logic                                                rstn,

    // Input/output boundary bank status.
    input  logic [1:0]                                          in_bank_full,
    input  logic [1:0]                                          out_bank_full,

    // Global pipeline phase.
    output logic                                                phase_sel,
    output logic                                                phase_active,
    output logic                                                in_consume_pulse,
    output logic                                                out_produce_pulse,

    // Input boundary buffer read interface.
    output logic                                                in_rd_en,
    output logic [((INPUT_SIZE>1)?$clog2(INPUT_SIZE):1)-1:0]    in_rd_addr,
    input  logic signed [ACT_W-1:0]                             in_rd_data,

    // Output boundary buffer write interface.
    output logic                                                out_wr_en,
    output logic [((OUTPUT_SIZE>1)?$clog2(OUTPUT_SIZE):1)-1:0]  out_wr_addr,
    output logic signed [ACT_W-1:0]                             out_wr_data
);

    localparam int NUM_LINEAR_CFG = `NUM_LINEAR;
    localparam int LEAK_SHIFT_CFG  = `LEAK_SHIFT;

    // -------------------------------------------------------------------------
    // Generated topology and scale tables.
    // Values are macros from scales.vh; Python only generates the list shape.
    // -------------------------------------------------------------------------
    localparam int N_TAB [0:NUM_LINEAR_CFG] = '{
        {n_tab_expr}
    };

    localparam int SX_TAB [0:NUM_LINEAR_CFG] = '{
        {sx_tab_expr}
    };

    localparam int SW_TAB [0:NUM_LINEAR_CFG-1] = '{
        {sw_tab_expr}
    };

    localparam int SB_TAB [0:NUM_LINEAR_CFG-1] = '{
        {sb_tab_expr}
    };

    localparam int SP_TAB [0:NUM_LINEAR_CFG-2] = '{
        {sp_tab_expr}
    };

    localparam int ACT_TAB [0:NUM_LINEAR_CFG-1] = '{
        {act_tab_expr}
    };

    localparam int LUTN_TAB [0:NUM_LINEAR_CFG-1] = '{
        {lutn_tab_expr}
    };

    logic [NUM_LINEAR_CFG-1:0] layer_done_pulse;
    logic                      start_phase_pulse;

    // -------------------------------------------------------------------------
    // Global phase controller.
    // phase_sel selects the input bank read by the first layer.
    // ~phase_sel selects the output bank written by the last layer.
    // -------------------------------------------------------------------------
    phase_ctrl #(
        .N_LAYERS(NUM_LINEAR_CFG)
    ) u_phase_ctrl (
        .clk              (clk),
        .rstn             (rstn),

        .in_bank_full     (in_bank_full),
        .out_bank_full    (out_bank_full),
        .layer_done_pulse (layer_done_pulse),

        .phase_sel        (phase_sel),
        .phase_active     (phase_active),
        .start_phase_pulse(start_phase_pulse),
        .in_consume_pulse (in_consume_pulse),
        .out_produce_pulse(out_produce_pulse)
    );

    // -------------------------------------------------------------------------
    // Generated intermediate ping-pong buffers.
    // One buffer is inserted between each pair of consecutive layers.
    // -------------------------------------------------------------------------
    {intermediate_buffers}

    // -------------------------------------------------------------------------
    // Generated layer instances.
    // Each instance uses its own generated layer module because the ROM module
    // names are fixed at elaboration time and cannot be built from LAYER_ID.
    // -------------------------------------------------------------------------
    {layer_instances}

endmodule

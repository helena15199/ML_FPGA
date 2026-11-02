// -----------------------------------------------------------------------------
// dsp.sv - Lattice Radiant / iCE40UP5K version using pmi_dsp
//
// Implements on each enabled clock cycle:
//   first_mac = 1 : P <= C + A*B
//   first_mac = 0 : P <= P + A*B
//
// IMPORTANT:
//   - Intended for the INT16 x INT16 -> INT32 Lattice MLP MAC path.
//   - Radiant does NOT accept the iCEcube2-only SB_MAC16 primitive directly.
//     pmi_dsp is the Radiant wrapper around the iCE40UP MAC16 DSP block.
//   - Add the Radiant PMI synthesis header from <Radiant>/ip/pmi to the
//     implementation sources, as required by the Radiant PMI flow.
// -----------------------------------------------------------------------------

module dsp #(
    parameter int A_WIDTH = 16,
    parameter int B_WIDTH = 16,
    parameter int P_WIDTH = 32
)(
    input  logic                         clk,
    input  logic                         rstn,
    input  logic                         ce,
    input  logic                         first_mac,
    input  logic signed [A_WIDTH-1:0]    A,
    input  logic signed [B_WIDTH-1:0]    B,
    input  logic signed [P_WIDTH-1:0]    C,
    output logic signed [P_WIDTH-1:0]    P
);

    // Native iCE40 UltraPlus DSP operand widths.
    wire [15:0] a16 = A[15:0];
    wire [15:0] b16 = B[15:0];

    // Registered DSP result.
    wire [31:0] p32;

    // Preserve the original controller semantics without adding an extra cycle:
    // first MAC adds the aligned bias C; subsequent MACs feed back the previous P.
    wire [31:0] addend = first_mac ? C[31:0] : p32;

    // The accumulator output reset is synchronous in the DSP block.
    // Force CE active while reset is asserted so that the reset is observed.
    wire dsp_ce = ce | ~rstn;

    wire co_unused;
    wire accumco_unused;
    wire signextout_unused;

    // Radiant wrapper for the iCE40UP MAC16 DSP block.
    pmi_dsp #(
        .NEG_TRIGGER                 ("0b0"),

        // Combinational inputs; no extra latency relative to the current FSM.
        .A_REG                       ("0b0"),
        .B_REG                       ("0b0"),
        .C_REG                       ("0b0"),
        .D_REG                       ("0b0"),

        // No multiplier pipeline registers for this first timing experiment.
        .TOP_8x8_MULT_REG            ("0b0"),
        .BOT_8x8_MULT_REG            ("0b0"),
        .PIPELINE_16x16_MULT_REG1     ("0b0"),
        .PIPELINE_16x16_MULT_REG2     ("0b0"),

        // Select registered outputs of the upper/lower adder sections.
        .TOPOUTPUT_SELECT             ("0b01"),
        .BOTOUTPUT_SELECT             ("0b01"),

        // Lower adder inputs come from the 16x16 multiplication result.
        .TOPADDSUB_LOWERINPUT         ("0b10"),
        .BOTADDSUB_LOWERINPUT         ("0b10"),

        // Upper adder inputs come from C (upper 16 bits) and D (lower 16 bits).
        .TOPADDSUB_UPPERINPUT         ("0b1"),
        .BOTADDSUB_UPPERINPUT         ("0b1"),

        // Propagate carry from the lower 16-bit adder into the upper 16-bit adder.
        .TOPADDSUB_CARRYSELECT        ("0b11"),
        .BOTADDSUB_CARRYSELECT        ("0b00"),

        .MODE_8x8                     ("0b0"),
        .A_SIGNED                     ("0b1"),
        .B_SIGNED                     ("0b1")
    ) u_pmi_dsp (
        .CLK        (clk),
        .CE         (dsp_ce),

        .A          (a16),
        .B          (b16),
        .C          (addend[31:16]),
        .D          (addend[15:0]),

        .AHOLD      (1'b0),
        .BHOLD      (1'b0),
        .CHOLD      (1'b0),
        .DHOLD      (1'b0),

        .IRSTTOP    (1'b0),
        .IRSTBOT    (1'b0),
        .ORSTTOP    (~rstn),
        .ORSTBOT    (~rstn),

        .OLOADTOP   (1'b0),
        .OLOADBOT   (1'b0),
        .ADDSUBTOP  (1'b0),
        .ADDSUBBOT  (1'b0),
        .OHOLDTOP   (1'b0),
        .OHOLDBOT   (1'b0),

        .CI         (1'b0),
        .ACCUMCI    (1'b0),
        .SIGNEXTIN  (1'b0),

        .O          (p32),
        .CO         (co_unused),
        .ACCUMCO    (accumco_unused),
        .SIGNEXTOUT (signextout_unused)
    );

    // Keep the original parameterized interface.
    always_comb begin
        P = '0;
        P[31:0] = p32;
    end

endmodule
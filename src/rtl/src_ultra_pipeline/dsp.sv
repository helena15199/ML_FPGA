// -----------------------------------------------------------------------------
// dsp.sv - V2
// Shared signed MAC/interpolation datapath.
// P_WIDTH is the physical DSP accumulator width (DSP_PW from scales.vh).
// Arithmetic overflow is normal two's-complement wrap; calibration must ensure
// that valid model data never relies on that wrap.
// -----------------------------------------------------------------------------

module dsp #(
    parameter int A_WIDTH = 16,
    parameter int B_WIDTH = 16,
    parameter int P_WIDTH = 32
)(
    input  logic                         clk,
    input  logic                         rstn,
    input  logic                         ce,
    input  logic                         first_mac,   // 1: P <= C + A*B; 0: P <= P + A*B
    input  logic signed [A_WIDTH-1:0]    A,
    input  logic signed [B_WIDTH-1:0]    B,
    input  logic signed [P_WIDTH-1:0]    C,
    output logic signed [P_WIDTH-1:0]    P
);

    (* use_dsp = "yes" *)
    logic signed [P_WIDTH-1:0] mult_w;

    assign mult_w = $signed(A) * $signed(B);

    always_ff @(posedge clk) begin
        if (ce) begin
            if (first_mac) P <= C + mult_w;
            else           P <= P + mult_w;
        end
    end

    wire _unused_rstn = rstn;

endmodule

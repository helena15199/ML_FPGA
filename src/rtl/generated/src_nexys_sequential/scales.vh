// -----------------------------------------------------------------------------
// Macro configuration (auto-generated)
// -----------------------------------------------------------------------------
`ifndef SCALES
`define SCALES

// V2 numerical widths
`define ACT_W       16
`define WEIGHT_W    16
`define BIAS_W      16
`define DSP_PW      48

// --- Topologie ---
`define NUM_LINEAR  3
`define N0  784
`define N1  64
`define N2  32
`define N3  10

// Activation codes:
// 0 = identity
// 1 = relu
// 2 = leaky_relu_pow2
// 3 = lut

// -----------------------------------------------------------------------------
// Model scale factors
//
// SXi : tensor scale at the input of layer i
//       and SX{NUM_LINEAR} = network output scale
// SWi : weight scale of layer i
// SBi : bias scale of layer i
// SPi : pre-activation scale of layer i
//       (present only for hidden layers)
// -----------------------------------------------------------------------------
// Couche 0
`define SX0 14
`define SW0 14
`define SB0 15
`define SP0 10

// Couche 1
`define SX1 10
`define SW1 15
`define SB1 15
`define SP1 10

// Couche 2
`define SX2 10
`define SW2 14
`define SB2 16

// Network output
`define SX3 9

// -----------------------------------------------------------------------------
// Activation type per linear layer
// -----------------------------------------------------------------------------
`define ACT0 2
`define ACT1 2
`define ACT2 0

// Use only if ACTi = 2
`define LEAK_SHIFT 6

// Use only if ACTi = 3
`define LUT_N0 0
`define LUT_N1 0
`define LUT_N2 0

`endif // SCALES

// -----------------------------------------------------------------------------
// Macro configuration (auto-generated)
// -----------------------------------------------------------------------------
`ifndef SCALES
`define SCALES

// V2 numerical widths (fallback example; generated scales.vh takes precedence)
`define ACT_W                16
`define WEIGHT_W             16
`define BIAS_W               16
`define DSP_PW               32

// --- Topologie ---
`define NUM_LINEAR  6
`define N0  6
`define N1  66
`define N2  50
`define N3  38
`define N4  28
`define N5  20
`define N6  2

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
`define SX0 15
`define SW0 16
`define SB0 17
`define SP0 16

// Couche 1
`define SX1 16
`define SW1 15
`define SB1 17
`define SP1 16

// Couche 2
`define SX2 16
`define SW2 14
`define SB2 17
`define SP2 16

// Couche 3
`define SX3 16
`define SW3 15
`define SB3 17
`define SP3 15

// Couche 4
`define SX4 15
`define SW4 15
`define SB4 18
`define SP4 15

// Couche 5
`define SX5 15
`define SW5 15
`define SB5 19

// Network output
`define SX6 15

// -----------------------------------------------------------------------------
// Activation type per linear layer
// -----------------------------------------------------------------------------
`define ACT0 2
`define ACT1 2
`define ACT2 2
`define ACT3 2
`define ACT4 2
`define ACT5 0

// Use only if ACTi = 2
`define LEAK_SHIFT 6

// Use only if ACTi = 3
`define LUT_N0 0
`define LUT_N1 0
`define LUT_N2 0
`define LUT_N3 0
`define LUT_N4 0
`define LUT_N5 0

`endif // SCALES

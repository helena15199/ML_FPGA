// -----------------------------------------------------------------------------
// rom_act_all.sv - V2
// Concatenated activation LUT ROM for the sequential engine.
// ACT_W is the single width used by network activations, LUT input words,
// LUT output words and the final network output.
//
// IMPORTANT: the next local address intentionally wraps modulo 2**lut_n.
// Python calibration must guarantee that the unsafe final positive segment is
// never reached; this RTL does not clamp or saturate it.
// -----------------------------------------------------------------------------

module rom_act_all #(
    parameter int ACT_W      = 16,
    parameter int DEPTH      = 1,
    parameter int ADDR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1
)(
    input  logic                         clk,
    input  logic                         en,
    input  logic [ADDR_WIDTH-1:0]        base_addr,
    input  logic [$clog2(ACT_W+1)-1:0]  lut_n,
    input  logic signed [ACT_W-1:0]      x,

    output logic signed [ACT_W-1:0]      ya,
    output logic signed [ACT_W-1:0]      yb,
    output logic [ACT_W-1:0]             frac
);

    localparam int LUTN_W = $clog2(ACT_W+1);
    localparam logic [LUTN_W-1:0] ACT_W_L = LUTN_W'(ACT_W);

    logic signed [ACT_W-1:0] mem [0:DEPTH-1];

    logic [ACT_W-1:0] x_u;
    logic [ACT_W-1:0] addr_mask;
    logic [ACT_W-1:0] frac_mask;
    logic [ACT_W-1:0] local_addr_a_w;
    logic [ACT_W-1:0] local_addr_b_w;
    logic [ACT_W-1:0] frac_w;

    integer shift_addr;
    integer addr_a_i;
    integer addr_b_i;

    // ROM contents are injected by the Python/TCL project generator.
    // The generated FPGA RTL contains explicit mem[index] assignments.
    initial begin
        mem[0] = 16'h0000;
    end

    always_comb begin
        x_u            = $unsigned(x);
        addr_mask      = '0;
        frac_mask      = '0;
        local_addr_a_w = '0;
        local_addr_b_w = '0;
        frac_w         = '0;
        shift_addr     = 0;
        addr_a_i       = 0;
        addr_b_i       = 0;

        if ((lut_n > 0) && (lut_n <= ACT_W_L)) begin
            shift_addr = ACT_W - lut_n;

            if (lut_n == ACT_W_L)
                addr_mask = {ACT_W{1'b1}};
            else
                addr_mask = ({{(ACT_W-1){1'b0}}, 1'b1} << lut_n) - {{(ACT_W-1){1'b0}}, 1'b1};

            local_addr_a_w = (x_u >> shift_addr) & addr_mask;
            local_addr_b_w = (local_addr_a_w + {{(ACT_W-1){1'b0}}, 1'b1}) & addr_mask;

            if (shift_addr <= 0) begin
                frac_w = '0;
            end else if (shift_addr >= ACT_W) begin
                frac_w = x_u;
            end else begin
                frac_mask = ({{(ACT_W-1){1'b0}}, 1'b1} << shift_addr) - {{(ACT_W-1){1'b0}}, 1'b1};
                frac_w = x_u & frac_mask;
            end

            addr_a_i = $unsigned(base_addr) + $unsigned(local_addr_a_w);
            addr_b_i = $unsigned(base_addr) + $unsigned(local_addr_b_w);
        end
    end

    always_ff @(posedge clk) begin
        if (en) begin
            ya   <= mem[addr_a_i];
            yb   <= mem[addr_b_i];
            frac <= frac_w;
        end
    end

endmodule



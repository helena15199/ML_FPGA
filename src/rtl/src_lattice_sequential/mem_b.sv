// -----------------------------------------------------------------------------
// mem_b.sv
// Simple synchronous single-port RAM used for activations
// data storage inside the MLP accelerator.
// Read and write operations are performed on the rising clock edge.
// Memory width and depth are fully parameterizable.
// -----------------------------------------------------------------------------
module mem_b #(
    parameter int ACT_W = 16,
    parameter int DEPTH = 64,
    parameter int AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1
)(
    input  logic                 clk,

    input  logic                 rd_en,
    input  logic [AW-1:0]        rd_addr,
    output logic signed [ACT_W-1:0] rd_data,

    input  logic                 wr_en,
    input  logic [AW-1:0]        wr_addr,
    input  logic signed [ACT_W-1:0] wr_data
);

    logic signed [ACT_W-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end

        if (rd_en) begin
            rd_data <= mem[rd_addr];
        end
    end

endmodule

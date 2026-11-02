// -----------------------------------------------------------------------------
// pp_buffer.sv
// Ping-pong buffer used between pipeline stages.
//
// The buffer contains two banks:
//   - bank A when bank_sel = 0
//   - bank B when bank_sel = 1
//
// One side can read from one bank while the other side writes to the other bank.
// Bank scheduling is handled outside this module by phase_ctrl / layer control.
// ----------------------------------------------------------------------------- 

module pp_buffer #(
    parameter int ACT_W = 16,
    parameter int DEPTH = 64,
    parameter int AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1
)(
    input  logic                     clk,

    input  logic                     rd_bank_sel, // 0=A, 1=B
    input  logic                     rd_en,
    input  logic [AW-1:0]            rd_addr,
    output logic signed [ACT_W-1:0]  rd_data,

    input  logic                     wr_bank_sel, // 0=A, 1=B
    input  logic                     wr_en,
    input  logic [AW-1:0]            wr_addr,
    input  logic signed [ACT_W-1:0]  wr_data
);

    logic signed [ACT_W-1:0] mem_a [0:DEPTH-1];
    logic signed [ACT_W-1:0] mem_b [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (wr_en) begin
            if (!wr_bank_sel) mem_a[wr_addr] <= wr_data;
            else              mem_b[wr_addr] <= wr_data;
        end

        if (rd_en) begin
            if (!rd_bank_sel) rd_data <= mem_a[rd_addr];
            else              rd_data <= mem_b[rd_addr];
        end
    end

endmodule

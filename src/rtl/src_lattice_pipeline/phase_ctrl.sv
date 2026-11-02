// -----------------------------------------------------------------------------
// phase_ctrl.sv
// Global phase controller for the layer pipeline.
//
// The pipeline uses ping-pong buffers at the input and output boundaries.
// For each phase:
//   - phase_sel selects the input bank to consume
//   - ~phase_sel selects the output bank to produce
//
// A new phase starts only when:
//   - the selected input bank contains a complete frame
//   - the corresponding output bank is free
//   - no phase is currently active
//
// The phase completes when all layers have asserted their done pulse.
// -----------------------------------------------------------------------------

module phase_ctrl #(
    parameter int N_LAYERS = 6
)(
    input  logic                  clk,
    input  logic                  rstn,

    input  logic [1:0]            in_bank_full,
    input  logic [1:0]            out_bank_full,
    input  logic [N_LAYERS-1:0]   layer_done_pulse,

    output logic                  phase_sel,        // 0: read bank A / write bank B, 1: read bank B / write bank A
    output logic                  phase_active,
    output logic                  start_phase_pulse,
    output logic                  in_consume_pulse,
    output logic                  out_produce_pulse
);

    logic [N_LAYERS-1:0] done_seen;
    logic                can_start;
    logic                all_done_now;

    assign can_start    = !phase_active && in_bank_full[phase_sel] && !out_bank_full[~phase_sel];
    assign all_done_now = &(done_seen | layer_done_pulse);

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            phase_sel         <= 1'b0;
            phase_active      <= 1'b0;
            start_phase_pulse <= 1'b0;
            in_consume_pulse  <= 1'b0;
            out_produce_pulse <= 1'b0;
            done_seen         <= '0;
        end else begin
            start_phase_pulse <= 1'b0;
            in_consume_pulse  <= 1'b0;
            out_produce_pulse <= 1'b0;

            if (!phase_active) begin
                done_seen <= '0;

                if (can_start) begin
                    start_phase_pulse <= 1'b1;
                    phase_active      <= 1'b1;
                end
            end else begin
                done_seen <= done_seen | layer_done_pulse;

                if (all_done_now) begin
                    phase_active      <= 1'b0;
                    in_consume_pulse  <= 1'b1;
                    out_produce_pulse <= 1'b1;
                    phase_sel         <= ~phase_sel;
                    done_seen         <= '0;
                end
            end
        end
    end

endmodule


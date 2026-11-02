`timescale 1ns/1ps
`include "scales.vh"

// Behavioral simulation of the supplied Ultra96 sequential SOC_MLP.
// Core latency: accepting input_consume_pulse -> final result RAM write
// (mlp_done_pulse sampled on a rising clock edge). AXI loading/draining excluded.
module TB_mlp_latency;
    parameter real CLK_PERIOD_NS = 10.0; // Ultra96: 100 MHz
    localparam int SAMPLES = 4;
    localparam int TIMEOUT_CYCLES = 100000;
    logic clk = 0;
    logic rstn = 0;
    always #(CLK_PERIOD_NS / 2.0) clk = ~clk;

    logic [15:0] s_axis_s2mm_tdata = 0;
    logic [1:0] s_axis_s2mm_tkeep = 2'b11;
    logic s_axis_s2mm_tlast = 0;
    logic s_axis_s2mm_tvalid = 0;
    wire s_axis_s2mm_tready;
    wire [15:0] m_axis_mm2s_tdata;
    wire [1:0] m_axis_mm2s_tkeep;
    wire m_axis_mm2s_tlast;
    logic m_axis_mm2s_tready = 1;
    wire m_axis_mm2s_tvalid;
    SOC_MLP dut (.*);

    logic signed [15:0] examples [0:SAMPLES-1][0:`N0-1];
    longint cycle = 0;
    longint core_start_cycle, axis_start_cycle, core_cycles;
    realtime core_start_ns, core_ns;
    int sample_id = 0, received = 0, completed = 0, fd;
    bit core_active = 0, core_finished = 0, axis_active = 0;

    // Sample controls before nonblocking assignments update the DUT state.
    // In particular, done marks the edge that commits the final RAM write.
    always @(posedge clk) begin
        cycle = cycle + 1;
        if (rstn) begin
            if (s_axis_s2mm_tvalid && s_axis_s2mm_tready && !axis_active) begin
                axis_active = 1;
                axis_start_cycle = cycle;
            end
            if (dut.input_consume_pulse) begin
                if (core_active) $fatal(1, "Overlapping MLP computations");
                core_active = 1;
                core_finished = 0;
                core_start_cycle = cycle;
                core_start_ns = $realtime;
            end
            if (dut.mlp_done_pulse) begin
                if (!core_active) $fatal(1, "MLP done without start");
                core_cycles = cycle - core_start_cycle;
                core_ns = $realtime - core_start_ns;
                core_active = 0;
                core_finished = 1;
                $display("SAMPLE %0d: MLP = %0d cycles, %0.3f ns = %0.6f us at %0.3f MHz",
                         sample_id, core_cycles, core_ns, core_ns/1000.0,
                         1000.0/CLK_PERIOD_NS);
            end
            if (m_axis_mm2s_tvalid && m_axis_mm2s_tready) begin
                if (!core_finished) $fatal(1, "Output before computation completed");
                if ($isunknown(m_axis_mm2s_tdata)) $fatal(1, "Unknown output data");
                if (m_axis_mm2s_tkeep !== 2'b11) $fatal(1, "Invalid TKEEP");
                if (m_axis_mm2s_tlast !== (received == `N6-1))
                    $fatal(1, "Invalid TLAST on output %0d", received);
                $display("  output[%0d] = %0d (0x%04h)", received,
                         $signed(m_axis_mm2s_tdata), m_axis_mm2s_tdata);
                received = received + 1;
                if (received == `N6) begin
                    $fdisplay(fd, "%0d,%0.6f,%0d,%0.3f,%0.6f,%0d,%0.6f",
                              sample_id, 1000.0/CLK_PERIOD_NS, core_cycles,
                              core_ns, core_ns/1000.0, cycle-axis_start_cycle,
                              (cycle-axis_start_cycle)*CLK_PERIOD_NS/1000.0);
                    $display("  AXIS first input -> last output = %0d cycles (includes adapters)",
                             cycle-axis_start_cycle);
                    completed = completed + 1;
                    received = 0;
                    axis_active = 0;
                end
            end
        end
        if (cycle > SAMPLES * TIMEOUT_CYCLES)
            $fatal(1, "Timeout: simulation did not complete");
    end

    task automatic send_word(input logic [15:0] value, input bit last);
        @(negedge clk);
        s_axis_s2mm_tdata = value;
        s_axis_s2mm_tlast = last;
        s_axis_s2mm_tvalid = 1;
        do @(posedge clk); while (!s_axis_s2mm_tready);
        @(negedge clk);
        s_axis_s2mm_tvalid = 0;
        s_axis_s2mm_tlast = 0;
    endtask

    initial begin
        if (`N0 != 6) $fatal(1, "Adapt the six-feature examples to this topology");
        // Raw signed INT16 words, already at the model input scale SX0.
        // These are timing stimuli, not a numerical-accuracy reference test.
        for (int i=0; i<`N0; i++) begin
            examples[0][i] = 0;
            examples[1][i] = 1000 * (i+1);
            examples[2][i] = (i%2 == 0) ? 2000*(i+1) : -2000*(i+1);
            examples[3][i] = (i%2 == 0) ? 16'sh7fff : 16'sh8000;
        end
        fd = $fopen("mlp_latency.csv", "w");
        if (!fd) $fatal(1, "Cannot write mlp_latency.csv");
        $fdisplay(fd, "sample,clock_mhz,core_cycles,core_ns,core_us,axis_cycles,axis_us");
        repeat (10) @(negedge clk);
        rstn = 1;
        repeat (3) @(negedge clk);
        for (int s=0; s<SAMPLES; s++) begin
            sample_id = s;
            $display("--- SAMPLE %0d ---", s);
            for (int i=0; i<`N0; i++)
                send_word(examples[s][i], i == `N0-1);
            while (completed <= s) @(negedge clk);
            repeat (3) @(negedge clk);
        end
        $fclose(fd);
        $display("PASS: %0d samples timed. Results: mlp_latency.csv", completed);
        $finish;
    end
endmodule

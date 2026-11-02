`timescale 1ns/1ps
`include "scales.vh"

module TB_soc_mlp_pipeline;

    localparam int DW                = `DATA_WIDTH;
    localparam int INPUT_SZ          = `N0;
    localparam int OUTPUT_SZ = `N6;

    parameter int NUM_REAL_SAMPLES   = 100;
    parameter int WARMUP_FRAMES      = `NUM_LINEAR - 1;
    parameter int NUM_FLUSH_SAMPLES  = WARMUP_FRAMES;
    parameter int NUM_STREAM_SAMPLES = NUM_REAL_SAMPLES + NUM_FLUSH_SAMPLES;

    localparam int NUM_INPUT_WORDS   = NUM_STREAM_SAMPLES * INPUT_SZ;
    localparam int NUM_EXPECT_WORDS  = NUM_REAL_SAMPLES   * OUTPUT_SZ;

    parameter int TIMEOUT_CYCLES = 2000000;
    parameter int CHECK_DATA     = 1;
    parameter int TOL_LSB        = 0;

    parameter string INPUT_MEM_FILE    = "tb_inputs_stream.mem";
    parameter string EXPECTED_MEM_FILE = "tb_expected_outputs.mem";

    logic         ap_clk;
    logic         axi_resetn;

    logic [15:0]  s_axis_s2mm_tdata;
    logic [1:0]   s_axis_s2mm_tkeep;
    logic         s_axis_s2mm_tlast;
    logic         s_axis_s2mm_tvalid;
    logic         s_axis_s2mm_tready;

    logic [15:0]  m_axis_mm2s_tdata;
    logic [1:0]   m_axis_mm2s_tkeep;
    logic         m_axis_mm2s_tlast;
    logic         m_axis_mm2s_tready;
    logic         m_axis_mm2s_tvalid;

    logic [15:0] stim_mem [0:NUM_INPUT_WORDS-1];
    logic [15:0] exp_mem  [0:NUM_EXPECT_WORDS-1];

    int total_errors;
    int total_checked_words;
    int timeout_counter;

    SOC_MLP dut (
        .clk              (ap_clk),
        .rstn          (axi_resetn),

        .s_axis_s2mm_tdata   (s_axis_s2mm_tdata),
        .s_axis_s2mm_tkeep   (s_axis_s2mm_tkeep),
        .s_axis_s2mm_tlast   (s_axis_s2mm_tlast),
        .s_axis_s2mm_tvalid  (s_axis_s2mm_tvalid),
        .s_axis_s2mm_tready  (s_axis_s2mm_tready),

        .m_axis_mm2s_tdata   (m_axis_mm2s_tdata),
        .m_axis_mm2s_tkeep   (m_axis_mm2s_tkeep),
        .m_axis_mm2s_tlast   (m_axis_mm2s_tlast),
        .m_axis_mm2s_tready  (m_axis_mm2s_tready),
        .m_axis_mm2s_tvalid  (m_axis_mm2s_tvalid)
    );

    always #5 ap_clk = ~ap_clk;

    function automatic int signed s16(input logic [15:0] x);
        s16 = $signed(x);
    endfunction

    function automatic int abs_int(input int x);
        abs_int = (x < 0) ? -x : x;
    endfunction

    task automatic reset_dut;
        begin
            ap_clk = 1'b0;
            axi_resetn = 1'b0;

            s_axis_s2mm_tdata  = '0;
            s_axis_s2mm_tkeep  = 2'b11;
            s_axis_s2mm_tlast  = 1'b0;
            s_axis_s2mm_tvalid = 1'b0;

            m_axis_mm2s_tready = 1'b1;

            repeat (10) @(posedge ap_clk);
            axi_resetn = 1'b1;
            repeat (5) @(posedge ap_clk);
        end
    endtask

    task automatic axis_send_word(input logic [15:0] data, input bit last_word);
        begin
            s_axis_s2mm_tdata  <= data;
            s_axis_s2mm_tkeep  <= 2'b11;
            s_axis_s2mm_tlast  <= last_word;
            s_axis_s2mm_tvalid <= 1'b1;

            do begin
                @(posedge ap_clk);
            end while (!(s_axis_s2mm_tvalid && s_axis_s2mm_tready));

            s_axis_s2mm_tvalid <= 1'b0;
            s_axis_s2mm_tlast  <= 1'b0;
            s_axis_s2mm_tdata  <= '0;

            @(posedge ap_clk);
        end
    endtask

    task automatic send_stream;
        int frame;
        int j;
        int base;
        begin
            for (frame = 0; frame < NUM_STREAM_SAMPLES; frame++) begin
                base = frame * INPUT_SZ;

                for (j = 0; j < INPUT_SZ; j++) begin
                    axis_send_word(stim_mem[base + j], j == INPUT_SZ-1);
                end

                if (frame < NUM_REAL_SAMPLES) begin
                    $display("[%0t] IN real frame %0d pushed", $time, frame);
                end else begin
                    $display("[%0t] IN flush frame %0d pushed", $time, frame);
                end
            end
        end
    endtask

    task automatic receive_stream;
        int rx_frame;
        int k;
        int sample_idx;
        int expected_base;
        int signed got;
        int signed exp;
        int signed diff;
        bit expected_last;
        begin
            rx_frame = 0;
            k = 0;

            while (rx_frame < NUM_STREAM_SAMPLES) begin
                @(posedge ap_clk);

                if (m_axis_mm2s_tvalid && m_axis_mm2s_tready) begin

                    expected_last = (k == OUTPUT_SZ-1);

                    if (m_axis_mm2s_tlast !== expected_last) begin
                        $display("ERROR TLAST: rx_frame=%0d word=%0d got=%0b expected=%0b",
                                 rx_frame, k, m_axis_mm2s_tlast, expected_last);
                        total_errors++;
                    end

                    if (rx_frame < WARMUP_FRAMES) begin
                        $display("[%0t] OUT warmup frame=%0d word=%0d data=%0d",
                                 $time, rx_frame, k, s16(m_axis_mm2s_tdata));
                    end else begin
                        sample_idx = rx_frame - WARMUP_FRAMES;

                        if (sample_idx < NUM_REAL_SAMPLES) begin
                            expected_base = sample_idx * OUTPUT_SZ;

                            got  = s16(m_axis_mm2s_tdata);
                            exp  = s16(exp_mem[expected_base + k]);
                            diff = got - exp;

                            $display("[%0t] OUT frame=%0d sample=%0d word=%0d got=%0d expected=%0d diff=%0d",
                                     $time, rx_frame, sample_idx, k, got, exp, diff);

                            if (CHECK_DATA && abs_int(diff) > TOL_LSB) begin
                                $display("ERROR DATA: sample=%0d output=%0d got=%0d expected=%0d diff=%0d",
                                         sample_idx, k, got, exp, diff);
                                total_errors++;
                            end

                            total_checked_words++;
                        end
                    end

                    if (k == OUTPUT_SZ-1) begin
                        k = 0;
                        rx_frame++;
                    end else begin
                        k++;
                    end
                end
            end
        end
    endtask

    always_ff @(posedge ap_clk or negedge axi_resetn) begin
        if (!axi_resetn) begin
            timeout_counter <= 0;
        end else begin
            timeout_counter <= timeout_counter + 1;

            if (timeout_counter > TIMEOUT_CYCLES) begin
                $fatal(1, "Timeout simulation.");
            end
        end
    end

    initial begin
        total_errors = 0;
        total_checked_words = 0;
        timeout_counter = 0;

        $display("============================================================");
        $display("TB SOC_MLP - stream avec flush pipeline");
        $display("INPUT_SZ            = %0d", INPUT_SZ);
        $display("OUTPUT_SZ           = %0d", OUTPUT_SZ);
        $display("NUM_REAL_SAMPLES    = %0d", NUM_REAL_SAMPLES);
        $display("NUM_LINEAR          = %0d", `NUM_LINEAR);
        $display("WARMUP_FRAMES       = %0d", WARMUP_FRAMES);
        $display("NUM_FLUSH_SAMPLES   = %0d", NUM_FLUSH_SAMPLES);
        $display("NUM_STREAM_SAMPLES  = %0d", NUM_STREAM_SAMPLES);
        $display("INPUT_MEM_FILE      = %s", INPUT_MEM_FILE);
        $display("EXPECTED_MEM_FILE   = %s", EXPECTED_MEM_FILE);
        $display("============================================================");

        $readmemh(INPUT_MEM_FILE, stim_mem);
        $readmemh(EXPECTED_MEM_FILE, exp_mem);

        reset_dut();

        fork
            send_stream();
            receive_stream();
        join

        $display("------------------------------------------------------------");
        $display("Simulation terminée.");
        $display("Frames warmup invalides : %0d", WARMUP_FRAMES);
        $display("Frames réelles vérifiées: %0d", NUM_REAL_SAMPLES);
        $display("Words vérifiés          : %0d", total_checked_words);
        $display("Total errors            : %0d", total_errors);
        $display("Décalage appliqué       : output_frame - %0d", WARMUP_FRAMES);
        $display("------------------------------------------------------------");

        if (total_errors == 0) begin
            $display("PASS");
        end else begin
            $display("FAIL");
            $fatal(1);
        end

        $finish;
    end

endmodule


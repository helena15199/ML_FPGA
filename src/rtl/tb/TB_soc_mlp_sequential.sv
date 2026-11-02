`include "scales.vh"

module TB_soc_mlp_sequential;

    localparam int DW          = `DATA_WIDTH;
    localparam int INPUT_SIZE  = `N0;
    localparam int OUTPUT_SIZE = `N6;

    parameter int N_SAMPLES = 100;
    parameter int TOL_LSB = 0;
    parameter int TIMEOUT_CYCLES = 200000;

    parameter string INPUT_MEM_FILE    = "tb_inputs_stream.mem";
    parameter string EXPECTED_MEM_FILE = "tb_expected_outputs.mem";
    parameter string RESULT_CSV_FILE   = "tb_soc_mlp_results.csv";

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

    logic [15:0] input_mem    [0:N_SAMPLES*INPUT_SIZE-1];
    logic [15:0] expected_mem [0:N_SAMPLES*OUTPUT_SIZE-1];

    int fd;
    int total_errors;
    int total_words;

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
            end while (!s_axis_s2mm_tready);

            s_axis_s2mm_tvalid <= 1'b0;
            s_axis_s2mm_tlast  <= 1'b0;
            s_axis_s2mm_tdata  <= '0;
        end
    endtask

    task automatic send_sample(input int sample_idx);
        int j;
        int base;
        begin
            base = sample_idx * INPUT_SIZE;
            for (j = 0; j < INPUT_SIZE; j++) begin
                axis_send_word(input_mem[base + j], (j == INPUT_SIZE-1));
            end
        end
    endtask

    task automatic receive_and_check_sample(input int sample_idx);
        int k;
        int cycles;
        int expected_base;
        int signed got;
        int signed exp;
        int signed diff;
        bit expected_last;
        begin
            k = 0;
            cycles = 0;
            expected_base = sample_idx * OUTPUT_SIZE;
            m_axis_mm2s_tready <= 1'b1;

            while (k < OUTPUT_SIZE) begin
                @(posedge ap_clk);
                cycles++;

                if (cycles > TIMEOUT_CYCLES) begin
                    $display("ERROR: timeout while waiting output. sample=%0d received=%0d/%0d", sample_idx, k, OUTPUT_SIZE);
                    $display("       m_valid=%0b m_ready=%0b m_last=%0b", m_axis_mm2s_tvalid, m_axis_mm2s_tready, m_axis_mm2s_tlast);
                    $fatal(1);
                end

                if (m_axis_mm2s_tvalid && m_axis_mm2s_tready) begin
                    got = s16(m_axis_mm2s_tdata);
                    exp = s16(expected_mem[expected_base + k]);
                    diff = got - exp;
                    expected_last = (k == OUTPUT_SIZE-1);

                    $fwrite(fd, "%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                        sample_idx, k, got, exp, diff, m_axis_mm2s_tlast, expected_last);

                    if (m_axis_mm2s_tlast !== expected_last) begin
                        $display("ERROR TLAST: sample=%0d output=%0d got_last=%0b expected_last=%0b",
                            sample_idx, k, m_axis_mm2s_tlast, expected_last);
                        total_errors++;
                    end

                    if (abs_int(diff) > TOL_LSB) begin
                        $display("ERROR DATA: sample=%0d output=%0d got=%0d expected=%0d diff=%0d",
                            sample_idx, k, got, exp, diff);
                        total_errors++;
                    end

                    total_words++;
                    k++;
                end
            end
        end
    endtask

    initial begin
        int sample;

        total_errors = 0;
        total_words = 0;

        $display("============================================================");
        $display("SOC_MLP testbench");
        $display("N_SAMPLES          = %0d", N_SAMPLES);
        $display("INPUT_SIZE         = %0d", INPUT_SIZE);
        $display("OUTPUT_SIZE        = %0d", OUTPUT_SIZE);
        $display("TOL_LSB            = %0d", TOL_LSB);
        $display("INPUT_MEM_FILE     = %s", INPUT_MEM_FILE);
        $display("EXPECTED_MEM_FILE  = %s", EXPECTED_MEM_FILE);
        $display("============================================================");

        $readmemh(INPUT_MEM_FILE, input_mem);
        $readmemh(EXPECTED_MEM_FILE, expected_mem);

        fd = $fopen(RESULT_CSV_FILE, "w");
        if (fd == 0) begin
            $fatal(1, "Cannot open result CSV file: %s", RESULT_CSV_FILE);
        end
        $fwrite(fd, "sample,output_idx,got_int16,expected_int16,diff_int16,got_tlast,expected_tlast\n");

        reset_dut();

        for (sample = 0; sample < N_SAMPLES; sample++) begin
            send_sample(sample);
            receive_and_check_sample(sample);

            if (((sample + 1) % 10) == 0 || sample == N_SAMPLES-1) begin
                $display("Processed %0d/%0d samples", sample + 1, N_SAMPLES);
            end
        end

        $fclose(fd);

        $display("============================================================");
        $display("SOC_MLP testbench finished");
        $display("Compared words : %0d", total_words);
        $display("Errors         : %0d", total_errors);
        $display("Result CSV     : %s", RESULT_CSV_FILE);
        $display("============================================================");

        if (total_errors == 0) begin
            $display("PASS");
        end else begin
            $display("FAIL");
            $fatal(1);
        end

        $finish;
    end

endmodule

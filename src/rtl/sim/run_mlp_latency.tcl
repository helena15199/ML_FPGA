# Keep interface and timing-control waveforms, without recording the ROM arrays.
log_wave /TB_mlp_latency/clk
log_wave /TB_mlp_latency/rstn
log_wave /TB_mlp_latency/dut/input_consume_pulse
log_wave /TB_mlp_latency/dut/mlp_busy
log_wave /TB_mlp_latency/dut/mlp_done_pulse
log_wave /TB_mlp_latency/dut/s_axis_s2mm_tvalid
log_wave /TB_mlp_latency/dut/s_axis_s2mm_tready
log_wave /TB_mlp_latency/dut/m_axis_mm2s_tvalid
log_wave /TB_mlp_latency/dut/m_axis_mm2s_tready
log_wave /TB_mlp_latency/dut/m_axis_mm2s_tlast
log_wave /TB_mlp_latency/dut/m_axis_mm2s_tdata
run all
quit

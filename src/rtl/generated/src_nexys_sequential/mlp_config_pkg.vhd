-- Auto-generated from mem_layout.json and scales.vh.
package mlp_config_pkg is
    constant ACT_W       : positive := 16;
    constant WEIGHT_W    : positive := 16;
    constant BIAS_W      : positive := 16;
    constant DSP_PW      : positive := 48;
    constant INPUT_SIZE  : positive := 784;
    constant OUTPUT_SIZE : positive := 10;
    constant NUM_LINEAR  : positive := 3;
    constant MAX_NEURONS : positive := 784;
    constant VEC_AW      : positive := 10;
end package mlp_config_pkg;

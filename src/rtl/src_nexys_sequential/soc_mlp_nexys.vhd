-- -----------------------------------------------------------------------------
-- soc_mlp_nexys.vhd
-- Top level for the Nexys board: UART link (VHDL) around the sequential MLP core
-- of the repository (SystemVerilog). Protocol, 115200 baud 8N1, one image at a time:
--   PC -> FPGA : INPUT_SIZE signed 16-bit little-endian words
--   FPGA -> PC : 1 class byte, then OUTPUT_SIZE signed 16-bit little-endian scores
-- -----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mlp_config_pkg.all;

entity soc_mlp_nexys is
    generic (
        CLK_FREQ  : positive := 100_000_000;
        BAUD_RATE : positive := 115200
    );
    port (
        clk_100mhz   : in  std_logic;
        cpu_resetn   : in  std_logic;
        uart_txd_in  : in  std_logic; -- PC to FPGA
        uart_rxd_out : out std_logic  -- FPGA to PC
    );
end entity soc_mlp_nexys;

architecture rtl of soc_mlp_nexys is

    constant UART_IDLE_TIMEOUT_CYCLES : positive := CLK_FREQ / 100; -- 10 ms timeout

    -- Ping-pong RAM: layer 0 writes B, layer 1 writes A, layer 2 writes B, ...
    function final_bank return std_logic is
    begin
        if NUM_LINEAR mod 2 = 1 then
            return '1';
        end if;
        return '0';
    end function;

    component mlp_sequential is
        generic (
            ACT_W          : integer;
            WEIGHT_W       : integer;
            BIAS_W         : integer;
            DSP_PW         : integer;
            INPUT_SIZE     : integer;
            OUTPUT_SIZE    : integer;
            NUM_LINEAR_CFG : integer;
            MAX_NEURONS    : integer;
            VEC_AW         : integer
        );
        port (
            clk                  : in  std_logic;
            rstn                 : in  std_logic;
            input_frame_full     : in  std_logic;
            output_busy          : in  std_logic;
            input_consume_pulse  : out std_logic;
            output_produce_pulse : out std_logic;
            vec_rd_en            : out std_logic;
            vec_rd_bank_sel      : out std_logic;
            vec_rd_addr          : out std_logic_vector(VEC_AW - 1 downto 0);
            vec_rd_data          : in  std_logic_vector(ACT_W - 1 downto 0);
            vec_wr_en            : out std_logic;
            vec_wr_bank_sel      : out std_logic;
            vec_wr_addr          : out std_logic_vector(VEC_AW - 1 downto 0);
            vec_wr_data          : out std_logic_vector(ACT_W - 1 downto 0);
            busy                 : out std_logic;
            done_pulse           : out std_logic
        );
    end component;

    component mem_a is
        generic (
            ACT_W : integer;
            DEPTH : integer;
            AW    : integer
        );
        port (
            clk     : in  std_logic;
            rd_en   : in  std_logic;
            rd_addr : in  std_logic_vector(AW - 1 downto 0);
            rd_data : out std_logic_vector(ACT_W - 1 downto 0);
            wr_en   : in  std_logic;
            wr_addr : in  std_logic_vector(AW - 1 downto 0);
            wr_data : in  std_logic_vector(ACT_W - 1 downto 0)
        );
    end component;

    component mem_b is
        generic (
            ACT_W : integer;
            DEPTH : integer;
            AW    : integer
        );
        port (
            clk     : in  std_logic;
            rd_en   : in  std_logic;
            rd_addr : in  std_logic_vector(AW - 1 downto 0);
            rd_data : out std_logic_vector(ACT_W - 1 downto 0);
            wr_en   : in  std_logic;
            wr_addr : in  std_logic_vector(AW - 1 downto 0);
            wr_data : in  std_logic_vector(ACT_W - 1 downto 0)
        );
    end component;

    subtype vec_addr_t is std_logic_vector(VEC_AW - 1 downto 0);
    subtype vec_data_t is std_logic_vector(ACT_W - 1 downto 0);

    -- Reset: button synchronized, then held for a short delay after release.
    signal btn_meta : std_logic := '0';
    signal btn_sync : std_logic := '0';
    signal rst_cnt  : unsigned(15 downto 0) := (others => '0');
    signal rstn     : std_logic := '0';

    signal rx_data  : std_logic_vector(7 downto 0);
    signal rx_valid : std_logic;
    signal tx_data  : std_logic_vector(7 downto 0);
    signal tx_valid : std_logic;
    signal tx_busy  : std_logic;

    signal input_accept_enable  : std_logic;
    signal input_frame_full     : std_logic;
    signal input_consume_pulse  : std_logic;
    signal output_produce_pulse : std_logic;
    signal output_busy          : std_logic;
    signal mlp_busy             : std_logic;

    signal uart_in_wr_bank_sel  : std_logic;
    signal uart_in_wr_en        : std_logic;
    signal uart_in_wr_addr      : vec_addr_t;
    signal uart_in_wr_data      : vec_data_t;

    signal mlp_vec_rd_en        : std_logic;
    signal mlp_vec_rd_bank_sel  : std_logic;
    signal mlp_vec_rd_addr      : vec_addr_t;
    signal mlp_vec_wr_en        : std_logic;
    signal mlp_vec_wr_bank_sel  : std_logic;
    signal mlp_vec_wr_addr      : vec_addr_t;
    signal mlp_vec_wr_data      : vec_data_t;

    signal uart_out_rd_bank_sel : std_logic;
    signal uart_out_rd_en       : std_logic;
    signal uart_out_rd_addr     : vec_addr_t;

    signal ram_rd_en            : std_logic;
    signal ram_rd_bank_sel      : std_logic;
    signal ram_rd_bank_sel_q    : std_logic := '0';
    signal ram_rd_addr          : vec_addr_t;
    signal ram_rd_data          : vec_data_t;
    signal ram_wr_en            : std_logic;
    signal ram_wr_bank_sel      : std_logic;
    signal ram_wr_addr          : vec_addr_t;
    signal ram_wr_data          : vec_data_t;

    signal mem_a_rd_en          : std_logic;
    signal mem_b_rd_en          : std_logic;
    signal mem_a_wr_en          : std_logic;
    signal mem_b_wr_en          : std_logic;
    signal mem_a_rd_data        : vec_data_t;
    signal mem_b_rd_data        : vec_data_t;

begin

    process (clk_100mhz)
    begin
        if rising_edge(clk_100mhz) then
            btn_meta <= cpu_resetn;
            btn_sync <= btn_meta;

            if btn_sync = '0' then
                rst_cnt <= (others => '0');
                rstn    <= '0';
            elsif rst_cnt /= x"FFFF" then
                rst_cnt <= rst_cnt + 1;
                rstn    <= '0';
            else
                rstn <= '1';
            end if;
        end if;
    end process;

    -- MEM_A / MEM_B arbitration.
    -- uart_input writes A when the MLP is not running.
    -- mlp_sequential reads/writes A/B while it is busy.
    -- uart_output reads the final bank when the MLP is done.
    input_accept_enable <= not mlp_busy and not output_busy;

    ram_rd_en       <= mlp_vec_rd_en       when mlp_busy = '1' else uart_out_rd_en;
    ram_rd_bank_sel <= mlp_vec_rd_bank_sel when mlp_busy = '1' else uart_out_rd_bank_sel;
    ram_rd_addr     <= mlp_vec_rd_addr     when mlp_busy = '1' else uart_out_rd_addr;

    ram_wr_en       <= mlp_vec_wr_en       when mlp_busy = '1' else uart_in_wr_en;
    ram_wr_bank_sel <= mlp_vec_wr_bank_sel when mlp_busy = '1' else uart_in_wr_bank_sel;
    ram_wr_addr     <= mlp_vec_wr_addr     when mlp_busy = '1' else uart_in_wr_addr;
    ram_wr_data     <= mlp_vec_wr_data     when mlp_busy = '1' else uart_in_wr_data;

    mem_a_rd_en <= ram_rd_en and not ram_rd_bank_sel;
    mem_b_rd_en <= ram_rd_en and ram_rd_bank_sel;
    mem_a_wr_en <= ram_wr_en and not ram_wr_bank_sel;
    mem_b_wr_en <= ram_wr_en and ram_wr_bank_sel;

    -- Synchronous RAM: store the requested bank during the rd_en cycle.
    process (clk_100mhz, rstn)
    begin
        if rstn = '0' then
            ram_rd_bank_sel_q <= '0';
        elsif rising_edge(clk_100mhz) then
            if ram_rd_en = '1' then
                ram_rd_bank_sel_q <= ram_rd_bank_sel;
            end if;
        end if;
    end process;

    ram_rd_data <= mem_b_rd_data when ram_rd_bank_sel_q = '1' else mem_a_rd_data;

    u_uart_rx : entity work.uart_rx
        generic map (
            CLK_FREQ  => CLK_FREQ,
            BAUD_RATE => BAUD_RATE
        )
        port map (
            clk        => clk_100mhz,
            rstn       => rstn,
            rx         => uart_txd_in,
            data_out   => rx_data,
            data_valid => rx_valid
        );

    u_uart_input : entity work.uart_input
        generic map (
            ACT_W               => ACT_W,
            INPUT_SIZE          => INPUT_SIZE,
            AW                  => VEC_AW,
            IDLE_TIMEOUT_CYCLES => UART_IDLE_TIMEOUT_CYCLES
        )
        port map (
            clk           => clk_100mhz,
            rstn          => rstn,
            uart_data     => rx_data,
            uart_valid    => rx_valid,
            accept_enable => input_accept_enable,
            consume_pulse => input_consume_pulse,
            frame_full    => input_frame_full,
            loading       => open,
            wr_bank_sel   => uart_in_wr_bank_sel,
            wr_en         => uart_in_wr_en,
            wr_addr       => uart_in_wr_addr,
            wr_data       => uart_in_wr_data
        );

    u_mem_a : mem_a
        generic map (
            ACT_W => ACT_W,
            DEPTH => MAX_NEURONS,
            AW    => VEC_AW
        )
        port map (
            clk     => clk_100mhz,
            rd_en   => mem_a_rd_en,
            rd_addr => ram_rd_addr,
            rd_data => mem_a_rd_data,
            wr_en   => mem_a_wr_en,
            wr_addr => ram_wr_addr,
            wr_data => ram_wr_data
        );

    u_mem_b : mem_b
        generic map (
            ACT_W => ACT_W,
            DEPTH => MAX_NEURONS,
            AW    => VEC_AW
        )
        port map (
            clk     => clk_100mhz,
            rd_en   => mem_b_rd_en,
            rd_addr => ram_rd_addr,
            rd_data => mem_b_rd_data,
            wr_en   => mem_b_wr_en,
            wr_addr => ram_wr_addr,
            wr_data => ram_wr_data
        );

    u_mlp_seq : mlp_sequential
        generic map (
            ACT_W          => ACT_W,
            WEIGHT_W       => WEIGHT_W,
            BIAS_W         => BIAS_W,
            DSP_PW         => DSP_PW,
            INPUT_SIZE     => INPUT_SIZE,
            OUTPUT_SIZE    => OUTPUT_SIZE,
            NUM_LINEAR_CFG => NUM_LINEAR,
            MAX_NEURONS    => MAX_NEURONS,
            VEC_AW         => VEC_AW
        )
        port map (
            clk                  => clk_100mhz,
            rstn                 => rstn,
            input_frame_full     => input_frame_full,
            output_busy          => output_busy,
            input_consume_pulse  => input_consume_pulse,
            output_produce_pulse => output_produce_pulse,
            vec_rd_en            => mlp_vec_rd_en,
            vec_rd_bank_sel      => mlp_vec_rd_bank_sel,
            vec_rd_addr          => mlp_vec_rd_addr,
            vec_rd_data          => ram_rd_data,
            vec_wr_en            => mlp_vec_wr_en,
            vec_wr_bank_sel      => mlp_vec_wr_bank_sel,
            vec_wr_addr          => mlp_vec_wr_addr,
            vec_wr_data          => mlp_vec_wr_data,
            busy                 => mlp_busy,
            done_pulse           => open
        );

    u_uart_output : entity work.uart_output
        generic map (
            ACT_W          => ACT_W,
            OUTPUT_SIZE    => OUTPUT_SIZE,
            AW             => VEC_AW,
            FINAL_BANK_SEL => final_bank
        )
        port map (
            clk           => clk_100mhz,
            rstn          => rstn,
            produce_pulse => output_produce_pulse,
            busy          => output_busy,
            done_pulse    => open,
            rd_bank_sel   => uart_out_rd_bank_sel,
            rd_en         => uart_out_rd_en,
            rd_addr       => uart_out_rd_addr,
            rd_data       => ram_rd_data,
            tx_data       => tx_data,
            tx_valid      => tx_valid,
            tx_busy       => tx_busy
        );

    u_uart_tx : entity work.uart_tx
        generic map (
            CLK_FREQ  => CLK_FREQ,
            BAUD_RATE => BAUD_RATE
        )
        port map (
            clk        => clk_100mhz,
            rstn       => rstn,
            data_in    => tx_data,
            data_valid => tx_valid,
            tx         => uart_rxd_out,
            busy       => tx_busy
        );

end architecture rtl;

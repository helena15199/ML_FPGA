-- -----------------------------------------------------------------------------
-- uart_input.vhd
-- Receives one image over UART as signed 16-bit little-endian words and writes
-- it into RAM bank A. A partial frame is discarded after IDLE_TIMEOUT_CYCLES
-- without any received byte.
-- -----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_input is
    generic (
        ACT_W               : positive := 16;
        INPUT_SIZE          : positive := 784;
        AW                  : positive := 10;
        IDLE_TIMEOUT_CYCLES : positive := 1_000_000
    );
    port (
        clk           : in  std_logic;
        rstn          : in  std_logic;

        uart_data     : in  std_logic_vector(7 downto 0);
        uart_valid    : in  std_logic;

        -- High only when RAM banks A/B are available.
        accept_enable : in  std_logic;

        -- Pulse from the MLP engine when the frame in RAM bank A is consumed.
        consume_pulse : in  std_logic;
        frame_full    : out std_logic;
        loading       : out std_logic;

        wr_bank_sel   : out std_logic;
        wr_en         : out std_logic;
        wr_addr       : out std_logic_vector(AW - 1 downto 0);
        wr_data       : out std_logic_vector(ACT_W - 1 downto 0)
    );
end entity uart_input;

architecture rtl of uart_input is

    signal wr_ptr     : unsigned(AW - 1 downto 0) := (others => '0');
    signal lsb_byte   : std_logic_vector(7 downto 0) := (others => '0');
    signal byte_phase : std_logic := '0'; -- 0: wait for LSB, 1: wait for MSB
    signal idle_cnt   : natural range 0 to IDLE_TIMEOUT_CYCLES := 0;

    signal frame_full_i : std_logic := '0';
    signal loading_i    : std_logic := '0';
    signal byte_fire    : std_logic;
    signal uart_word    : std_logic_vector(15 downto 0);

begin

    frame_full <= frame_full_i;
    loading    <= loading_i;

    uart_word <= uart_data & lsb_byte;
    byte_fire <= uart_valid and accept_enable and not frame_full_i;

    wr_bank_sel <= '0'; -- bank A
    wr_en       <= byte_fire and byte_phase;
    wr_addr     <= std_logic_vector(wr_ptr);
    wr_data     <= uart_word(ACT_W - 1 downto 0);

    process (clk, rstn)
    begin
        if rstn = '0' then
            frame_full_i <= '0';
            loading_i    <= '0';
            wr_ptr       <= (others => '0');
            lsb_byte     <= (others => '0');
            byte_phase   <= '0';
            idle_cnt     <= 0;
        elsif rising_edge(clk) then
            if consume_pulse = '1' then
                frame_full_i <= '0';
            end if;

            if byte_fire = '1' then
                idle_cnt <= 0;

                if byte_phase = '0' then
                    -- First byte of the 16-bit word: LSB.
                    lsb_byte   <= uart_data;
                    byte_phase <= '1';
                    loading_i  <= '1';
                else
                    -- Second byte of the 16-bit word: MSB.
                    byte_phase <= '0';

                    if to_integer(wr_ptr) = INPUT_SIZE - 1 then
                        frame_full_i <= '1';
                        loading_i    <= '0';
                        wr_ptr       <= (others => '0');
                    else
                        wr_ptr    <= wr_ptr + 1;
                        loading_i <= '1';
                    end if;
                end if;
            elsif loading_i = '1' and frame_full_i = '0' then
                -- A frame has started but is not complete.
                if idle_cnt >= IDLE_TIMEOUT_CYCLES - 1 then
                    loading_i  <= '0';
                    wr_ptr     <= (others => '0');
                    lsb_byte   <= (others => '0');
                    byte_phase <= '0';
                    idle_cnt   <= 0;
                else
                    idle_cnt <= idle_cnt + 1;
                end if;
            else
                idle_cnt <= 0;
            end if;
        end if;
    end process;

end architecture rtl;

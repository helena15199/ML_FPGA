-- -----------------------------------------------------------------------------
-- uart_rx.vhd
-- Simple UART receiver with 8 data bits, no parity, and one stop bit.
-- The asynchronous RX input is synchronized before being sampled.
-- -----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_rx is
    generic (
        CLK_FREQ  : positive := 100_000_000;
        BAUD_RATE : positive := 115200
    );
    port (
        clk        : in  std_logic;
        rstn       : in  std_logic;
        rx         : in  std_logic;
        data_out   : out std_logic_vector(7 downto 0);
        data_valid : out std_logic
    );
end entity uart_rx;

architecture rtl of uart_rx is

    constant CLKS_PER_BIT : positive := (CLK_FREQ + (BAUD_RATE / 2)) / BAUD_RATE;

    type state_t is (S_IDLE, S_START, S_DATA, S_STOP);
    signal state : state_t := S_IDLE;

    signal clk_cnt    : natural range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_idx    : natural range 0 to 7 := 0;
    signal data_shift : std_logic_vector(7 downto 0) := (others => '0');

    -- Synchronize the external serial input into the clk domain.
    signal rx_meta : std_logic := '1';
    signal rx_sync : std_logic := '1';

begin

    process (clk, rstn)
    begin
        if rstn = '0' then
            rx_meta <= '1';
            rx_sync <= '1';
        elsif rising_edge(clk) then
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end if;
    end process;

    process (clk, rstn)
    begin
        if rstn = '0' then
            state      <= S_IDLE;
            clk_cnt    <= 0;
            bit_idx    <= 0;
            data_shift <= (others => '0');
            data_out   <= (others => '0');
            data_valid <= '0';
        elsif rising_edge(clk) then
            data_valid <= '0';

            case state is
                when S_IDLE =>
                    clk_cnt <= 0;
                    bit_idx <= 0;

                    -- UART idle line is high. A start bit is detected as a low level.
                    if rx_sync = '0' then
                        state <= S_START;
                    end if;

                when S_START =>
                    -- Validate the start bit at its midpoint.
                    if clk_cnt = CLKS_PER_BIT / 2 then
                        clk_cnt <= 0;

                        if rx_sync = '0' then
                            state <= S_DATA;
                        else
                            state <= S_IDLE;
                        end if;
                    else
                        clk_cnt <= clk_cnt + 1;
                    end if;

                when S_DATA =>
                    if clk_cnt = CLKS_PER_BIT - 1 then
                        clk_cnt             <= 0;
                        data_shift(bit_idx) <= rx_sync;

                        if bit_idx = 7 then
                            bit_idx <= 0;
                            state   <= S_STOP;
                        else
                            bit_idx <= bit_idx + 1;
                        end if;
                    else
                        clk_cnt <= clk_cnt + 1;
                    end if;

                when S_STOP =>
                    if clk_cnt = CLKS_PER_BIT - 1 then
                        clk_cnt <= 0;
                        state   <= S_IDLE;

                        -- Accept the frame only if the stop bit is high.
                        if rx_sync = '1' then
                            data_out   <= data_shift;
                            data_valid <= '1';
                        end if;
                    else
                        clk_cnt <= clk_cnt + 1;
                    end if;
            end case;
        end if;
    end process;

end architecture rtl;

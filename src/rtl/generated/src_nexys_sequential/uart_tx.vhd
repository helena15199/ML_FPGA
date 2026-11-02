-- -----------------------------------------------------------------------------
-- uart_tx.vhd
-- Simple UART transmitter with 8 data bits, no parity, and one stop bit.
-- The frame is serialized LSB first when data_valid is asserted.
-- -----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_tx is
    generic (
        CLK_FREQ  : positive := 100_000_000;
        BAUD_RATE : positive := 115200
    );
    port (
        clk        : in  std_logic;
        rstn       : in  std_logic;
        data_in    : in  std_logic_vector(7 downto 0);
        data_valid : in  std_logic;
        tx         : out std_logic;
        busy       : out std_logic
    );
end entity uart_tx;

architecture rtl of uart_tx is

    constant CLKS_PER_BIT : positive := (CLK_FREQ + (BAUD_RATE / 2)) / BAUD_RATE;

    type state_t is (S_IDLE, S_SEND);
    signal state : state_t := S_IDLE;

    signal clk_cnt     : natural range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_idx     : natural range 0 to 9 := 0;
    signal frame_shift : std_logic_vector(9 downto 0) := (others => '1'); -- stop, data, start

begin

    process (clk, rstn)
    begin
        if rstn = '0' then
            state       <= S_IDLE;
            clk_cnt     <= 0;
            bit_idx     <= 0;
            frame_shift <= (others => '1');
            tx          <= '1';
            busy        <= '0';
        elsif rising_edge(clk) then
            case state is
                when S_IDLE =>
                    tx      <= '1';
                    busy    <= '0';
                    clk_cnt <= 0;
                    bit_idx <= 0;

                    if data_valid = '1' then
                        frame_shift <= '1' & data_in & '0';
                        tx          <= '0'; -- start bit
                        busy        <= '1';
                        state       <= S_SEND;
                    end if;

                when S_SEND =>
                    busy <= '1';

                    if clk_cnt = CLKS_PER_BIT - 1 then
                        clk_cnt <= 0;

                        if bit_idx = 9 then
                            bit_idx <= 0;
                            tx      <= '1';
                            busy    <= '0';
                            state   <= S_IDLE;
                        else
                            bit_idx <= bit_idx + 1;
                            tx      <= frame_shift(bit_idx + 1);
                        end if;
                    else
                        clk_cnt <= clk_cnt + 1;
                    end if;
            end case;
        end if;
    end process;

end architecture rtl;

-- -----------------------------------------------------------------------------
-- uart_output.vhd
-- Reads the final MLP scores, finds the predicted class, then sends over UART:
--   byte 0            : predicted class (index of the highest score)
--   bytes 1 to 2*N    : the N scores as signed 16-bit little-endian words
-- On a tie the lowest index wins, like numpy.argmax.
-- -----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_output is
    generic (
        ACT_W          : positive := 16;
        OUTPUT_SIZE    : positive := 10;
        AW             : positive := 10;
        FINAL_BANK_SEL : std_logic := '1'
    );
    port (
        clk           : in  std_logic;
        rstn          : in  std_logic;

        -- Pulse from the MLP engine when the last layer has written its scores.
        produce_pulse : in  std_logic;
        busy          : out std_logic;
        done_pulse    : out std_logic;

        rd_bank_sel   : out std_logic;
        rd_en         : out std_logic;
        rd_addr       : out std_logic_vector(AW - 1 downto 0);
        rd_data       : in  std_logic_vector(ACT_W - 1 downto 0);

        tx_data       : out std_logic_vector(7 downto 0);
        tx_valid      : out std_logic;
        tx_busy       : in  std_logic
    );
end entity uart_output;

architecture rtl of uart_output is

    type state_t is (
        S_IDLE,
        S_SCAN_FETCH,
        S_SCAN_NEXT,
        S_SEND_CLASS,
        S_WAIT_CLASS,
        S_FETCH,
        S_SEND_LSB,
        S_WAIT_LSB,
        S_SEND_MSB,
        S_WAIT_MSB,
        S_NEXT
    );
    signal state : state_t := S_IDLE;

    signal result_full : std_logic := '0';
    signal ptr         : unsigned(AW - 1 downto 0) := (others => '0');
    signal data_reg    : signed(ACT_W - 1 downto 0) := (others => '0');
    signal best_score  : signed(ACT_W - 1 downto 0) := (others => '0');
    signal best_class  : unsigned(7 downto 0) := (others => '0');
    signal tx_word     : std_logic_vector(15 downto 0);
    signal is_last     : boolean;

begin

    busy <= '0' when result_full = '0' and state = S_IDLE else '1';

    -- Scores narrower than 16 bits are sign-extended.
    tx_word <= std_logic_vector(resize(data_reg, 16));
    is_last <= to_integer(ptr) = OUTPUT_SIZE - 1;

    process (state, result_full, ptr, tx_busy, tx_word, best_class)
    begin
        rd_en       <= '0';
        rd_bank_sel <= FINAL_BANK_SEL;
        rd_addr     <= std_logic_vector(ptr);
        tx_data     <= (others => '0');
        tx_valid    <= '0';

        case state is
            when S_IDLE =>
                if result_full = '1' then
                    rd_en   <= '1';
                    rd_addr <= (others => '0');
                end if;

            when S_SCAN_NEXT | S_NEXT =>
                rd_en   <= '1';
                rd_addr <= std_logic_vector(ptr + 1);

            when S_SEND_CLASS =>
                if tx_busy = '0' then
                    tx_data  <= std_logic_vector(best_class);
                    tx_valid <= '1';
                end if;

            when S_WAIT_CLASS =>
                -- Restart reading from the first score.
                if tx_busy = '0' then
                    rd_en   <= '1';
                    rd_addr <= (others => '0');
                end if;

            when S_SEND_LSB =>
                if tx_busy = '0' then
                    tx_data  <= tx_word(7 downto 0);
                    tx_valid <= '1';
                end if;

            when S_SEND_MSB =>
                if tx_busy = '0' then
                    tx_data  <= tx_word(15 downto 8);
                    tx_valid <= '1';
                end if;

            when others =>
                null;
        end case;
    end process;

    process (clk, rstn)
    begin
        if rstn = '0' then
            state       <= S_IDLE;
            result_full <= '0';
            ptr         <= (others => '0');
            data_reg    <= (others => '0');
            best_score  <= (others => '0');
            best_class  <= (others => '0');
            done_pulse  <= '0';
        elsif rising_edge(clk) then
            done_pulse <= '0';

            if produce_pulse = '1' then
                result_full <= '1';
            end if;

            case state is
                when S_IDLE =>
                    ptr <= (others => '0');
                    if result_full = '1' then
                        state <= S_SCAN_FETCH;
                    end if;

                -- The RAM is synchronous: rd_en was asserted on the previous cycle.
                when S_SCAN_FETCH =>
                    if ptr = 0 or signed(rd_data) > best_score then
                        best_score <= signed(rd_data);
                        best_class <= resize(ptr, 8);
                    end if;

                    if is_last then
                        state <= S_SEND_CLASS;
                    else
                        state <= S_SCAN_NEXT;
                    end if;

                when S_SCAN_NEXT =>
                    ptr   <= ptr + 1;
                    state <= S_SCAN_FETCH;

                when S_SEND_CLASS =>
                    if tx_busy = '0' then
                        state <= S_WAIT_CLASS;
                    end if;

                when S_WAIT_CLASS =>
                    if tx_busy = '0' then
                        ptr   <= (others => '0');
                        state <= S_FETCH;
                    end if;

                when S_FETCH =>
                    data_reg <= signed(rd_data);
                    state    <= S_SEND_LSB;

                when S_SEND_LSB =>
                    if tx_busy = '0' then
                        state <= S_WAIT_LSB;
                    end if;

                when S_WAIT_LSB =>
                    if tx_busy = '0' then
                        state <= S_SEND_MSB;
                    end if;

                when S_SEND_MSB =>
                    if tx_busy = '0' then
                        state <= S_WAIT_MSB;
                    end if;

                when S_WAIT_MSB =>
                    if tx_busy = '0' then
                        if is_last then
                            result_full <= '0';
                            done_pulse  <= '1';
                            state       <= S_IDLE;
                        else
                            state <= S_NEXT;
                        end if;
                    end if;

                when S_NEXT =>
                    ptr   <= ptr + 1;
                    state <= S_FETCH;
            end case;
        end if;
    end process;

end architecture rtl;

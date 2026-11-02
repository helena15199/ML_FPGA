-- -----------------------------------------------------------------------------
-- tb_soc_mlp_nexys.vhd
-- Sends images over UART exactly like the PC and compares the 21 received bytes
-- with the vectors exported by the Python quantization. Baud rate is raised to
-- keep the simulation short. Add both .mem files to the Vivado simulation sources.
-- -----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;

library std;
use std.textio.all;

use work.mlp_config_pkg.all;

entity tb_soc_mlp_nexys is
    generic (
        INPUT_FILE    : string   := "tb_inputs_stream.mem";
        EXPECTED_FILE : string   := "tb_expected_outputs.mem";
        NUM_IMAGES    : positive := 3;
        TB_BAUD_RATE  : positive := 2_000_000
    );
end entity tb_soc_mlp_nexys;

architecture sim of tb_soc_mlp_nexys is

    constant CLK_FREQ    : positive := 100_000_000;
    constant CLK_PERIOD  : time     := 10 ns;
    constant BIT_PERIOD  : time     := CLK_PERIOD * ((CLK_FREQ + (TB_BAUD_RATE / 2)) / TB_BAUD_RATE);

    signal clk      : std_logic := '0';
    signal resetn   : std_logic := '0';
    signal to_dut   : std_logic := '1';
    signal from_dut : std_logic;
    signal running  : boolean   := true;

begin

    clk <= not clk after CLK_PERIOD / 2 when running else '0';

    dut : entity work.soc_mlp_nexys
        generic map (
            CLK_FREQ  => CLK_FREQ,
            BAUD_RATE => TB_BAUD_RATE
        )
        port map (
            clk_100mhz   => clk,
            cpu_resetn   => resetn,
            uart_txd_in  => to_dut,
            uart_rxd_out => from_dut
        );

    stimulus : process
        file inputs   : text open read_mode is INPUT_FILE;
        file expected : text open read_mode is EXPECTED_FILE;

        variable text_line   : line;
        variable word        : std_logic_vector(15 downto 0);
        variable received    : std_logic_vector(7 downto 0);
        variable score       : signed(15 downto 0);
        variable best_score  : signed(15 downto 0);
        variable best_class  : natural;
        variable class_byte  : natural;
        variable errors      : natural := 0;

        procedure send_byte (value : in std_logic_vector(7 downto 0)) is
        begin
            to_dut <= '0';
            wait for BIT_PERIOD;
            for i in 0 to 7 loop
                to_dut <= value(i);
                wait for BIT_PERIOD;
            end loop;
            to_dut <= '1';
            wait for BIT_PERIOD;
        end procedure;

        procedure receive_byte (value : out std_logic_vector(7 downto 0)) is
        begin
            wait until from_dut = '0';
            wait for BIT_PERIOD * 1.5;
            for i in 0 to 7 loop
                value(i) := from_dut;
                wait for BIT_PERIOD;
            end loop;
            assert from_dut = '1' report "Missing UART stop bit" severity error;
        end procedure;
    begin
        wait for 100 ns;
        resetn <= '1';
        wait for CLK_PERIOD * 65600; -- reset release delay of the top level

        for image in 0 to NUM_IMAGES - 1 loop
            for pixel in 0 to INPUT_SIZE - 1 loop
                readline(inputs, text_line);
                hread(text_line, word);
                send_byte(word(7 downto 0));
                send_byte(word(15 downto 8));
            end loop;

            receive_byte(received);
            class_byte := to_integer(unsigned(received));

            best_class := 0;
            for output_index in 0 to OUTPUT_SIZE - 1 loop
                readline(expected, text_line);
                hread(text_line, word);
                score := signed(word);
                if output_index = 0 or score > best_score then
                    best_score := score;
                    best_class := output_index;
                end if;

                receive_byte(received);
                if received /= word(7 downto 0) then
                    errors := errors + 1;
                    report "Image " & integer'image(image) & ": wrong LSB for score "
                        & integer'image(output_index) severity error;
                end if;
                receive_byte(received);
                if received /= word(15 downto 8) then
                    errors := errors + 1;
                    report "Image " & integer'image(image) & ": wrong MSB for score "
                        & integer'image(output_index) severity error;
                end if;
            end loop;

            if class_byte /= best_class then
                errors := errors + 1;
                report "Image " & integer'image(image) & ": class " & integer'image(class_byte)
                    & ", expected " & integer'image(best_class) severity error;
            end if;
            report "Image " & integer'image(image) & " done: class " & integer'image(class_byte);
            wait for BIT_PERIOD * 2;
        end loop;

        if errors = 0 then
            report "TEST PASSED: " & integer'image(NUM_IMAGES) & " images" severity note;
        else
            report "TEST FAILED: " & integer'image(errors) & " errors" severity error;
        end if;
        running <= false;
        wait;
    end process;

end architecture sim;

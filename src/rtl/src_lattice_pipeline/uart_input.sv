// -----------------------------------------------------------------------------
// uart_input.sv
// Receives an input vector over UART as signed 16-bit words.
//
// uart_rx handles the UART byte protocol.
// uart_input only handles the application framing:
//   - one input frame = exactly INPUT_SIZE words
//   - one word = two UART bytes, LSB first
//   - the useful payload is kept in the LSBs and resized to ACT_W bits
//   - input RAM bank availability follows the same phase/bank_full interface
//
// This module assumes ACT_W <= 16.
// -----------------------------------------------------------------------------

module uart_input #(
    parameter int ACT_W      = 16,
    parameter int INPUT_SIZE = 64,
    parameter int AW         = (INPUT_SIZE > 1) ? $clog2(INPUT_SIZE) : 1
)(
    input  logic                 clk,
    input  logic                 rstn,

    // Byte interface from uart_rx.
    input  logic [7:0]           rx_data,
    input  logic                 rx_valid,

    // Same pipeline control interface.
    input  logic                 phase_sel,
    input  logic                 phase_active,
    input  logic                 consume_pulse,
    output logic [1:0]           bank_full,
    output logic                 loading,

    // Write interface to the vector RAM bank.
    output logic                 wr_bank_sel,
    output logic                 wr_en,
    output logic [AW-1:0]        wr_addr,
    output logic signed [ACT_W-1:0] wr_data
);

    typedef enum logic [1:0] {
        S_IDLE,
        S_BYTE_LSB,
        S_BYTE_MSB
    } state_t;

    state_t state;

    logic [AW-1:0]      wr_ptr;
    logic               load_bank_reg;
    logic [7:0]         lsb_reg;
    logic signed [15:0] rx_word;

    logic               have_free_bank;
    logic               pick_bank;
    logic               word_fire;

    assign rx_word = $signed({rx_data, lsb_reg});

    generate
        if (ACT_W == 16) begin : g_uart_rx_same_width
            assign wr_data = rx_word;
        end else begin : g_uart_rx_truncate
            assign wr_data = $signed(rx_word[ACT_W-1:0]);
        end
    endgenerate

    // Same bank policy as the pipeline input side:
    // - during an active phase, the pipeline consumes one bank, so load the
    //   opposite bank if it is free;
    // - outside a phase, choose any free bank, prioritizing phase_sel.
    always_comb begin
        if (phase_active) begin
            pick_bank      = ~phase_sel;
            have_free_bank = !bank_full[~phase_sel];
        end else begin
            pick_bank      = !bank_full[phase_sel] ? phase_sel : ~phase_sel;
            have_free_bank = !bank_full[phase_sel] || !bank_full[~phase_sel];
        end
    end

    assign word_fire   = (state == S_BYTE_MSB) && rx_valid;
    assign wr_bank_sel = load_bank_reg;
    assign wr_en       = word_fire;
    assign wr_addr     = wr_ptr;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state         <= S_IDLE;
            bank_full     <= 2'b00;
            loading       <= 1'b0;
            wr_ptr        <= '0;
            load_bank_reg <= 1'b0;
            lsb_reg       <= 8'h00;
        end else begin
            if (consume_pulse) begin
                bank_full[phase_sel] <= 1'b0;
            end

            unique case (state)
                S_IDLE: begin
                    loading <= 1'b0;
                    wr_ptr  <= '0;

                    if (rx_valid && have_free_bank) begin
                        load_bank_reg <= pick_bank;
                        lsb_reg       <= rx_data;
                        loading       <= 1'b1;
                        state         <= S_BYTE_MSB;
                    end
                end

                S_BYTE_LSB: begin
                    if (rx_valid) begin
                        lsb_reg <= rx_data;
                        state   <= S_BYTE_MSB;
                    end
                end

                S_BYTE_MSB: begin
                    if (rx_valid) begin
                        if ($unsigned(wr_ptr) == (INPUT_SIZE-1)) begin
                            bank_full[load_bank_reg] <= 1'b1;
                            loading                  <= 1'b0;
                            wr_ptr                   <= '0;
                            state                    <= S_IDLE;
                        end else begin
                            wr_ptr <= wr_ptr + AW'(1);
                            state  <= S_BYTE_LSB;
                        end
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

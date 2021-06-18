#Clock signal 100M
create_clock -period 10.000 -name sys_clk_pin -add [get_ports clk]


`include "CPU.v"
`include "clock.v"

// run the CPU
module CPU_tb();

	clock clockunit(clk, 1'b1);
    cpu CPU_PRO (clk, 1'b0);
	
endmodule
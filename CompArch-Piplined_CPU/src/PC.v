// 32-bit PC
// data input width: 1 32-bit value (branch addr, PC+4, or jump addr)
// data output width: 1 32-bit addr


`define loaded

module PC  (
	input [31:0] PC_in,
    input clk, reset,
    
	output reg [31:0] PC_out
);
	initial begin
		PC_out <= 0;
	end

	
	always @(posedge clk) begin
		
		if(reset==1'b1)
			PC_out <= 0;
		else
			PC_out <= PC_in;
	end

endmodule
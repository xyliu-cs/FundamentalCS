module RegisterFile(BusA, BusB, BusW, RA, RB, RW, WrEn, clk, rst);

//Inputs: Read Register 1, Read Register 2, Write Register (5bits), Write Data (32bits), Write Enable

	input [4:0] RA, RB, RW;  //read addr A, read addr B, write addr
	input [31:0] BusW;  //data to write
	input WrEn;  //write enable
	input clk, rst;
	output [31:0] BusA, BusB;  //data to read
	
	reg [31:0] REG_MEM [31:0];

	integer i, j;
	
	initial begin //initialize all register values to 0
		for(i = 0; i < 32; i = i + 1) begin
				REG_MEM[i] <= 0;
		end
		// BusA = 32'b0;
		// BusB = 32'b0;
	end
			
	always @(posedge clk) begin
		if (rst) begin
			for(j = 0; j < 32; j = j + 1)
				REG_MEM[j] <= 0;
		end

		else if ((WrEn) && (RW))  //$0 must remain unchanged as 0, so RW cannot be 0
			REG_MEM[RW] <= BusW;		
	end

	assign BusA = REG_MEM[RA];
 	assign BusB = REG_MEM[RB]; 

endmodule

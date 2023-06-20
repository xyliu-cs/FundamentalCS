// MEM/WB stage register
// update content & output updated content at rising edge

// 1. WB control signal
//input RegWrite_in, [1:0]MemtoReg_in;
//output RegWrite_out, [1:0]MemtoReg_out;

// 2. data content
// input [31:0] read_data_in, aluout_in, PC_plus_4_in, Ins_In,
// input [4:0] EX_MEM_3_to_1_regAddr_in,
// output [31:0] read_data_out, aluout_out, PC_plus_4_out, Ins_out,
// output [4:0] EX_MEM_3_to_1_regAddr_out 

module MEM_WB (
	input RegWrite_in, 
	input [1:0] MemtoReg_in,
	input [31:0] read_data_in, aluout_in, PC_plus_4_in, Ins_In,
	input [4:0] EX_MEM_3_to_1_regAddr_in,
	input clk, rst,

	output RegWrite_out, 
	output [1:0] MemtoReg_out,
	output [31:0] read_data_out, aluout_out, PC_plus_4_out, Ins_out,
	output [4:0] EX_MEM_3_to_1_regAddr_out

);
	
	reg RegWrite_out;
	reg [1:0] MemtoReg_out;
	reg [31:0] read_data_out, aluout_out, PC_plus_4_out, Ins_out;
	reg [4:0] EX_MEM_3_to_1_regAddr_out;


	initial begin
		RegWrite_out <= 1'b0;
		MemtoReg_out <= 2'b00;
		read_data_out <= 32'b0;
		aluout_out <= 32'b0;
		PC_plus_4_out <= 32'b0;
		Ins_out <= 32'b0;
		EX_MEM_3_to_1_regAddr_out <= 5'b0;
	end
	
	always @(posedge clk)
	begin
		if (rst == 1'b1) begin
			RegWrite_out <= 1'b0;
			MemtoReg_out <= 2'b00;
			read_data_out <= 32'b0;
			aluout_out <= 32'b0;
			PC_plus_4_out <= 32'b0;
			EX_MEM_3_to_1_regAddr_out <= 5'b0;
			Ins_out <= 32'b0;
		end
		else begin
			RegWrite_out <= RegWrite_in;
			MemtoReg_out <= MemtoReg_in;
			read_data_out <= read_data_in;
			aluout_out <= aluout_in;
			EX_MEM_3_to_1_regAddr_out <= EX_MEM_3_to_1_regAddr_in;
			PC_plus_4_out <= PC_plus_4_in;
			Ins_out <= Ins_In;
		end
	end
	
endmodule


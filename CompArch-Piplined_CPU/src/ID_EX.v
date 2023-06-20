// 1. ID_EX Bank 

// 2. WB control signal
//input RegWrite_in, MemtoReg_in;
//output RegWrite_out, MemtoReg_out;

// 3. MEM control signal
//input Branch_in, MemRead_in, MemWrite_in, Jump_in;
//output Branch_out, MemRead_out, MemWrite_out, Jump_out;

// 4. addresses & instruction
// input PC_plus4_in, Ins_In;
// output PC_plus4_out, Ins_out;

// 5. data content
//input [31:0] reg_read_data_1_in, reg_read_data_2_in;
//output [31:0] reg_read_data_1_out, reg_read_data_2_out;

// 6. reg content
//input [4:0] IF_ID_RegisterRs_in, IF_ID_RegisterRt_in, IF_ID_RegisterRd_in;
//output [4:0] IF_ID_RegisterRs_out, IF_ID_RegisterRt_out, IF_ID_RegisterRd_out;

// general signal
// rst: async; set all register content to 0
//

module ID_EX (

	input Branch_in, MemRead_in, MemWrite_in, 
	input RegWrite_in, 
	input [1:0] RegDst_in, MemtoReg_in, Jump_in,
	input [31:0] PC_plus4_in, Ins_In,
	input [31:0] reg_read_data_1_in, reg_read_data_2_in,
	input [4:0] IF_ID_RegisterRs_in, IF_ID_RegisterRt_in, IF_ID_RegisterRd_in,

	input clk, rst,
	
	output RegWrite_out, 
	output Branch_out, MemRead_out, MemWrite_out, 

	output [1:0] RegDst_out, MemtoReg_out, Jump_out,
	output [31:0] PC_plus4_out, Ins_out,
	output [31:0] reg_read_data_1_out, reg_read_data_2_out,
	output [4:0] ID_EX_RegisterRs_out, ID_EX_RegisterRt_out, ID_EX_RegisterRd_out

);

	reg RegWrite_out;
	reg Branch_out, MemRead_out, MemWrite_out;
	reg [1:0] RegDst_out, MemtoReg_out,  Jump_out;

	reg [31:0] PC_plus4_out, Ins_out;

	reg [31:0] reg_read_data_1_out, reg_read_data_2_out;
	reg [4:0] ID_EX_RegisterRs_out, ID_EX_RegisterRt_out, ID_EX_RegisterRd_out;


	initial begin
		RegWrite_out <= 1'b0;
		RegDst_out <= 2'b00;
		MemtoReg_out <= 2'b00;			
		Branch_out <= 1'b0;
		Jump_out <= 2'b00;			
		MemRead_out <= 1'b0;
		MemWrite_out <= 1'b0;

		PC_plus4_out <= 32'b0;
		Ins_out <= 32'b0;

		reg_read_data_1_out <= 32'b0;
		reg_read_data_2_out <= 32'b0;

		ID_EX_RegisterRs_out <= 5'b0;
		ID_EX_RegisterRt_out <= 5'b0;
		ID_EX_RegisterRd_out <= 5'b0;
	end


	always @(posedge clk) begin
		
		if (rst == 1'b1) begin
			RegWrite_out <= 1'b0;
			RegDst_out <= 2'b00;
			MemtoReg_out <= 2'b00;			
			Branch_out <= 1'b0;
			Jump_out <= 2'b00;			
			MemRead_out <= 1'b0;
			MemWrite_out <= 1'b0;

			PC_plus4_out <= 32'b0;
			Ins_out <= 32'b0;

			reg_read_data_1_out <= 32'b0;
			reg_read_data_2_out <= 32'b0;

			ID_EX_RegisterRs_out <= 5'b0;
			ID_EX_RegisterRt_out <= 5'b0;
			ID_EX_RegisterRd_out <= 5'b0;
		
		end

		else begin

			RegWrite_out <= RegWrite_in;
			MemtoReg_out <= MemtoReg_in;
			Branch_out <= Branch_in;
			MemRead_out <= MemRead_in;
			MemWrite_out <= MemWrite_in;
			Jump_out <= Jump_in;
			RegDst_out <= RegDst_in;

			PC_plus4_out <= PC_plus4_in;
			Ins_out <= Ins_In;

			reg_read_data_1_out <= reg_read_data_1_in;
			reg_read_data_2_out <= reg_read_data_2_in;

			ID_EX_RegisterRs_out <= IF_ID_RegisterRs_in;
			ID_EX_RegisterRt_out <= IF_ID_RegisterRt_in;
			ID_EX_RegisterRd_out <= IF_ID_RegisterRd_in;

		end	
		
	end	
	
endmodule
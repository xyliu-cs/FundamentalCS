
//Outputs: branch_addr (32bits)
module branch_calculator(In1_instruction, In2_pc_plus_4, branch_addr);
	
	input [31:0] In1_instruction;
	input [31:0] In2_pc_plus_4;
	reg [15:0] immediate;
	reg [31:0] sign_extended;
	
	output reg [31:0] branch_addr;
	
	always @(*) begin
		immediate = In1_instruction[15:0];
		sign_extended = {{16{immediate[15]}}, immediate};
		branch_addr = (sign_extended << 2) + In2_pc_plus_4;
	end
endmodule




// Outputs: Jump Address 32bits (j/jal)

module jump_calculator(In1_instruction, In2_pc_plus_4, Jump_Address);

	input [31:0] In1_instruction;
	input [31:0] In2_pc_plus_4;
	
	output reg [31:0] Jump_Address;
	reg [27:0] temp;
	
	always @(In1_instruction) begin
		temp = In1_instruction[25:0] << 2'b10;
		// $display("In1_instruction:%b", In1_instruction);
		// $display("In1_instruction[25:0]:%b", In1_instruction[25:0]);
		// $display("temp:                 %b", temp);
		Jump_Address = {In2_pc_plus_4[31:28],temp};
		// $display("j to:         %d", Jump_Address);
    	// $display("-----------------------------------");
	end
endmodule


// Outputs: jr address calculator
module jr_calculator(rs_value, jr_addr);

	input [31:0] rs_value;	
	output reg [31:0] jr_addr;

	always @(rs_value) begin
		jr_addr = rs_value;
	end
endmodule
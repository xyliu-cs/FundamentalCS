module MUX_writeReg_addr (rt_addr, rd_addr, out_addr, RegDst);
input [4:0] rt_addr, rd_addr;
input [1:0] RegDst;
output reg [4:0] out_addr;

always @(rt_addr, rd_addr, RegDst) begin
    case (RegDst)
    0: out_addr <= rt_addr;
    1: out_addr <= rd_addr;
    2: out_addr <= 31;      // for jal instruction, write to $31 register
    endcase
end
endmodule


module MUX_writeReg_value (PC_4, MEM_out, ALU_out, MemToReg, value_to_write);
input [31:0] PC_4, MEM_out, ALU_out;
// MemToReg must be two bits because it is 0 for R format (ALU), 1 for lw/sw (Memory), 2 if jal (PC+4)
input [1:0] MemToReg;
output reg [31:0] value_to_write;

always @(PC_4, MEM_out, ALU_out, MemToReg) begin
    case (MemToReg)
    0: value_to_write <= ALU_out;
    1: value_to_write <= MEM_out;
    2: value_to_write <= PC_4;
    endcase
end
endmodule

//Mux to select the next PC address
module PC_input_mux_4_to_1(pc_plus_4, branch_addr_in, j_addr_in, jr_addr_in, branch_and_zero_sig, jump_sig, out);

	input [31:0] pc_plus_4, branch_addr_in, j_addr_in, jr_addr_in;
	input branch_and_zero_sig;
    input [1:0] jump_sig;
	output reg [31:0] out;

	initial begin
		out <= 0;		
	end
	
	always @(*) begin
		if (branch_and_zero_sig == 1)
			out <= branch_addr_in;
		// j/jal	
		else if (jump_sig == 1)
			out <= j_addr_in;
		// jr
		else if (jump_sig == 2)
			out <= jr_addr_in;
		// the above 3 conditions cannot occur simultaneously 
		else
			out <= pc_plus_4;			
			
	end

endmodule

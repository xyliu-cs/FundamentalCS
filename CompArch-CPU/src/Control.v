// Control Unit

// RegDst must be two bits because it is 1 if R format (rd), 0 if lw/sw (rt), and 2 if jal ($31),

// MemToReg must be two bits because it is 0 for R format (ALU), 1 for lw/sw (Memory), 2 if jal (PC+4), 

// Jump must be two bits because 0 for not jump, 1 for j/jal, 2 for jr.

// DON'T CARE FOR ANYTHING ELSE

module Control(opcode, funct, RegDst, MemWrite, MemRead, Branch, Jump, MemToReg, RegWrite);
	input [5:0] opcode, funct;
	output reg MemWrite, MemRead, Branch, RegWrite;
	output reg [1:0] RegDst, MemToReg, Jump;

	initial begin
		MemWrite <= 0; 
		MemRead <= 0;
		Branch <= 0; 
		RegWrite <= 0;
		RegDst <= 2'b0; 
		MemToReg <= 2'b00; 
		Jump <= 2'b0;
	end
// 7 signals

	always @(*) begin
		case (opcode)
			// R-type opcode 0x00
			6'b000000: begin
				// jr
				if (funct == 6'b001000) begin
					RegWrite <= 0;
					RegDst <= 0; // don't care
					MemToReg <= 2'b00; // don't care
					
					MemWrite <= 0;
					MemRead <= 0; 					
					
					Branch <= 0;
					Jump <= 2'b10;

				end

				else begin
					RegWrite <= 1'b1;
					RegDst <= 2'b01;
					MemToReg <= 2'b00;
					
					MemWrite <= 0;
					MemRead <= 0;
					
					Branch <= 0;
					Jump <= 0;
					
					
				end

			end

			// lw opcode 0x23
			6'b100011: begin
				RegWrite <= 1'b1;
				RegDst <= 0;
				MemToReg <= 2'b00;
				
				MemWrite <= 0;
				MemRead <= 1'b1;
				
				Branch <= 0;
				Jump <= 0;



			end

			// sw opcode 0x2b
			6'b101011: begin
				RegWrite <= 0;				
				RegDst <= 0; // don't care
				MemToReg <= 2'b01; // don't care				
				
				MemWrite <= 1;				
				MemRead <= 0;
				
				Branch <= 0;				
				Jump <= 0;


			end

			// beq opcode 0x04, bne opcode 0x05
			6'b000100, 6'b000101: begin
			    RegWrite <= 0;				
				RegDst <= 0; // don't care
				MemToReg <= 2'b00; // don't care				
			    
				MemWrite <= 0;
			    MemRead <= 0;
			    
				Branch <= 1;
			    Jump <= 0;


			end

			// j opcode 0x02
			6'b000010: begin
			    RegWrite <= 0;				
				RegDst <= 0;
				MemToReg <= 2'b00;
			    
				MemWrite <= 0;
			    MemRead <= 0;
			    
				Branch <= 0;
			    Jump <= 1;

			end
			
			// Jal opcode 0x03
			6'b000011: begin
			    RegWrite <= 1;				
				RegDst <= 2'b10;
				MemToReg <= 2'b10;
			    
				MemWrite <= 0;
			    MemRead <= 0;
			    
				Branch <= 0;
			    Jump <= 2'b01;

			end

			// addi opcode 0x08, addiu 0x09, andi 0x0C, ori 0x0D, xori 0x0E
			6'b001000, 6'b001001, 6'b001100, 6'b001101, 6'b001110: begin
			    
				RegWrite <= 1;
				MemToReg <= 2'b00;  // 0 is ALU
			    RegDst <= 0;   // 0 is rt
			    
				MemWrite <= 0;
			    MemRead <= 0;
			    
				Branch <= 0;
			    Jump <= 0;

			end
			

			
		endcase
		
	end
endmodule

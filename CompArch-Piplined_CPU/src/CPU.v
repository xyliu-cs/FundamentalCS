`ifndef loaded
`define loaded
//MODULES
`include "PC.v"
`include "InstructionRAM.v"
`include "Register.v"
`include "ALU.v"
`include "MainMemory.v"

//CONTROL
`include "Control.v"
`include "branch_jump_calc.v"
`include "multiplexers.v"


//PIPELINE REGISTERS
`include "IF_ID.v"
`include "ID_EX.v"
`include "EX_MEM.v"
`include "MEM_WB.v"
`endif

module cpu (clk, rst);

    input clk, rst;

    /* ----------------Updated at WB stage---------------- */
    wire [31:0] value_to_write;
	
    /* ----------------Updated at MEM stage---------------- */
	wire [31:0] branch_address, j_address, jr_address;


    // IF output wires
    wire [31:0] PC_in;
	wire [31:0] instruction_out, PC_out, IF_ID_instruction, IF_ID_PC_plus4;
    wire [31:0] PC_PLUS_FOUR;
    assign PC_PLUS_FOUR = PC_out + 4;                       // ATTENTION: PC = PC + 4 !!!


	PC_input_mux_4_to_1 PC_input_mux_4_to_1_UnitX(
        /*  ALL INPUTS ARE FROM THE MEM STAGE
            PAY ATTENTION     */
		.pc_plus_4(PC_PLUS_FOUR),  // from the second stage 
		.branch_addr_in(branch_address),
		.j_addr_in(j_address),
		.jr_addr_in(jr_address),
		.jump_sig(MEM_Jump),
		.branch_and_zero_sig(MEM_Branch & MEM_zeroflag),

        // out
		.out(PC_in)
	);

    /* ----------------IF stage---------------- */
	
	PC PC_Unit0 (
		.PC_in(PC_in), 
		.clk(clk), 
		.reset(rst),  
		.PC_out(PC_out)
	);
	
	InstructionRAM InstructionRAM_Unit1 (
		.CLOCK(clk), 
		.RESET(rst),
		.ENABLE(1'b1),
		.FETCH_ADDRESS(PC_out>>2),
		.DATA(instruction_out)
	);

    IF_ID IF_ID_Unit2 (
        // in
        .InsIn(instruction_out), 
        .PC_4_In(PC_out),                        
        // out
        .InsOut(IF_ID_instruction), 
        .PC_4_out(IF_ID_PC_plus4), 

        .clk(clk), 
        .reset(rst)
    );
	

    /* ----------------ID stage---------------- */

    // Control output wires
    wire [1:0] ID_RegDst, ID_MemToReg, ID_Jump;
    wire ID_RegWrite, ID_MemWrite, ID_MemRead, ID_Branch;
    // rs, rt value
    wire [31:0] reg_read_data_1, reg_read_data_2;

    wire [31:0] ID_EX_instruction, ID_EX_PC_plus4;
    wire [4:0] rs_addr, rt_addr, rd_addr;

    // bank output wires
    wire [1:0] EX_RegDst, EX_MemToReg, EX_Jump;
    wire EX_RegWrite, EX_MemWrite, EX_MemRead, EX_Branch;
    wire [4:0] EX_rs_addr, EX_rt_addr, EX_rd_addr;
    wire [31:0] EX_reg_read_data_1, EX_reg_read_data_2;

    


	Control Control_Unit3 (
        // in
		.opcode(IF_ID_instruction[31:26]),
		.funct(IF_ID_instruction[5:0]), 

        //out
		.RegDst(ID_RegDst), 
		.MemWrite(ID_MemWrite), 
		.MemRead(ID_MemRead), 
		.Branch(ID_Branch), 
		.Jump(ID_Jump), 
		.MemToReg(ID_MemToReg), 
		.RegWrite(ID_RegWrite)

	);
    
	RegisterFile RegisterFile_Unit4(
        //in
        .RA(IF_ID_instruction[25:21]), //rs address
		.RB(IF_ID_instruction[20:16]), //rt address


        //out
		.BusA(reg_read_data_1), // rs value
		.BusB(reg_read_data_2), // rt value

        //in
		.BusW(value_to_write),       // alu_result, memory_result, PC+4, to be declared
		.RW(WB_writeReg_addr),       // the addr of the register to write, should be from the MUX_writeReg_addr, to be declared
		.WrEn(WB_RegWrite),           // from RegWrite, to be assigned 
		
        .clk(clk), 
		.rst(rst)
	);

    ID_EX ID_EX_Unit5(
        // control signals input
        .Branch_in(ID_Branch), .MemRead_in(ID_MemRead), .MemWrite_in(ID_MemWrite), .RegWrite_in(ID_RegWrite),
        .RegDst_in(ID_RegDst), .MemtoReg_in(ID_MemToReg), .Jump_in(ID_Jump),

        .PC_plus4_in(IF_ID_PC_plus4), .Ins_In(IF_ID_instruction), 
        .reg_read_data_1_in(reg_read_data_1), .reg_read_data_2_in(reg_read_data_2),

		.IF_ID_RegisterRs_in(IF_ID_instruction[25:21]), 
        .IF_ID_RegisterRt_in(IF_ID_instruction[20:16]), 
		.IF_ID_RegisterRd_in(IF_ID_instruction[15:11]),

        //out

        //change wire names
	    .Branch_out(EX_Branch), .MemRead_out(EX_MemRead), .MemWrite_out(EX_MemWrite), .RegWrite_out(EX_RegWrite), 
        .RegDst_out(EX_RegDst), .MemtoReg_out(EX_MemToReg), .Jump_out(EX_Jump),
        // change wire names
        .PC_plus4_out(ID_EX_PC_plus4), .Ins_out(ID_EX_instruction),
        .reg_read_data_1_out(EX_reg_read_data_1), .reg_read_data_2_out(EX_reg_read_data_2),
        //new wire assigned
	    .ID_EX_RegisterRs_out(EX_rs_addr), .ID_EX_RegisterRt_out(EX_rt_addr), .ID_EX_RegisterRd_out(EX_rd_addr),

        .clk(clk), 
		.rst(rst)

    );

    /* ----------------EX stage---------------- */

    // EX stage wires
    wire [31:0] ALU_out;
    wire [4:0] writeReg_addr;
    wire zeroflag;
    wire [31:0] EX_MEM_PC_plus4, EX_MEM_instruction;

    // bank ouput new wires
    wire [1:0] MEM_MemToReg, MEM_Jump;
    wire MEM_RegWrite, MEM_MemWrite, MEM_MemRead, MEM_Branch;
    wire [31:0] MEM_ALU_out;
    wire [4:0] MEM_writeReg_addr;
    wire MEM_zeroflag;
    wire [31:0] MEM_reg_read_data_1, MEM_reg_read_data_2;


    alu ALU_Unit6(
        // in
        .instruction(ID_EX_instruction), .regA(EX_reg_read_data_1), .regB(EX_reg_read_data_2),
        // out
        .result(ALU_out), .zeroflag(zeroflag)
    );

    MUX_writeReg_addr MUX1_Unit7(
        //in
        .rt_addr(EX_rt_addr), .rd_addr(EX_rd_addr), .RegDst(EX_RegDst),
        //out
        .out_addr(writeReg_addr)
    );



    EX_MEM EX_MEM_Unit8(
        clk, rst,
        EX_RegWrite, 
        EX_MemToReg, EX_Jump,
        EX_Branch, EX_MemRead, EX_MemWrite, 
        ID_EX_PC_plus4, ID_EX_instruction,
        zeroflag,
        ALU_out, EX_reg_read_data_1, EX_reg_read_data_2,   
        writeReg_addr,
        MEM_RegWrite,
        MEM_MemToReg, MEM_Jump, 
	    MEM_Branch, MEM_MemRead, MEM_MemWrite,
        EX_MEM_PC_plus4, EX_MEM_instruction,
        MEM_zeroflag, 
        MEM_ALU_out, 
        MEM_reg_read_data_1, MEM_reg_read_data_2,
        MEM_writeReg_addr
    );

    /* ----------------MEM stage---------------- */

    // MEM stage wires
    
    wire [31:0] MEM_out;
    wire [31:0] MEM_WB_PC_plus4;
    wire WB_RegWrite;
    wire [1:0] WB_MemtoReg; //????
    wire [1:0] testout;

	wire [31:0] WB_MEM_out, WB_ALU_out, MEM_WB_instruction;
    wire [4:0] WB_writeReg_addr;

    MainMemory MainMemory_Unit9 (
        // in
        .CLOCK(clk), .RESET(rst), .ENABLE(1'b1), 
        .FETCH_ADDRESS(MEM_ALU_out>>2), .EDIT_SERIAL({MEM_MemWrite, MEM_ALU_out[31:0]>>2, MEM_reg_read_data_2}), 
        //out
        .DATA(MEM_out)
    );

    branch_calculator Branch_Unit10 (
        //in
        .In1_instruction(EX_MEM_instruction), .In2_pc_plus_4(EX_MEM_PC_plus4),
        //out
        .branch_addr(branch_address)                                        // NOTICE!!! DECALRED AT LINE 36, trace back 
    );

    // NOTICE!!! DECALRED AT LINE 36, trace back
    jump_calculator Jump_Unit11 (EX_MEM_instruction, EX_MEM_PC_plus4, j_address);    

    jr_calculator Jr_Unit12 (
        // int
        .rs_value(MEM_reg_read_data_1),
        //out
        .jr_addr(jr_address)                                                // NOTICE!!! DECALRED AT LINE 36, trace back 
    );
    
    // test testmodule (MEM_MemToReg, testout, clk);

    MEM_WB MEM_WB_Unit13 (
        MEM_RegWrite, MEM_MemToReg, MEM_out, MEM_ALU_out, EX_MEM_PC_plus4, EX_MEM_instruction, MEM_writeReg_addr, clk, rst,
        WB_RegWrite, testout, WB_MEM_out, WB_ALU_out, MEM_WB_PC_plus4, MEM_WB_instruction, WB_writeReg_addr
    );


    /* ----------------WB stage---------------- */
    
    // WB stage wires

    // DECLARED AT LINE 31, TRACE BACK
    MUX_writeReg_value MUX_writeReg_value_Unit14 (MEM_WB_PC_plus4, WB_MEM_out, WB_ALU_out, testout, value_to_write);
                                                                  
    


integer num = 0;
integer aa, of;
integer num_iter = 0;
integer flag = 0;

always @(negedge clk) begin

    #900;

    num = num + 1;

	$display("Clock cycles: %d", num);
    // // $display("PC_in:         %d", PC_in);
    $display("PC_out:         %d", PC_out);
    // $display("instructions:   %b", instruction_out);
    // $display("PC_PLUS_FOUR:  %d", PC_PLUS_FOUR);
    // $display("IF_ID_PC_plus4: %d", IF_ID_PC_plus4);
    // $display("ID_EX_PC_plus4: %d", ID_EX_PC_plus4);
    // $display("EX_MEM_PC_plus4: %d", EX_MEM_PC_plus4);
    // $display("MEM_WB_PC_plus4: %d", MEM_WB_PC_plus4);
    $display("-----------------------------------");

    
    


    if (num > 10000) begin
        aa=0;
        of = $fopen("./output.txt", "w");
        while(aa<512) begin
            $fwrite(of, "%b\n", MainMemory_Unit9.DATA_RAM[aa]);
            aa=aa+1;
        end   
        $display("Finished by reach the 10000 cycles protection limit");
	    $finish;
        end

    if (MEM_WB_instruction == 32'b11111111111111111111111111111111) begin
        aa=0;
        of = $fopen("./output.txt", "w");
        while(aa<512) begin
            $display("%b", MainMemory_Unit9.DATA_RAM[aa]);
            $fwrite(of, "%b\n", MainMemory_Unit9.DATA_RAM[aa]);
            aa=aa+1;
        end
        $display("Finished by reach the 32'hffffffff. Total lines: %d.", aa);
        
	    $finish;      
    end


end


endmodule


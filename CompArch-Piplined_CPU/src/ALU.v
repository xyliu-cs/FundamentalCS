module alu(instruction, regA, regB, result, zeroflag);

input signed [31:0] instruction, regA, regB; // the address of regA is 00000, the address of regB is 00001
output reg[31:0] result;
output reg zeroflag; // the first bit is zero flag, the second bit is negative flag, the third bit is overflow flag.


reg [5:0] opcode, func;
reg [4:0] reg_shift;
reg signed [4:0] shift_amount;
reg signed [15:0] immediate;

reg signed [31:0] rs, rt, reg_result;
reg signed [32:0] reg_result33;
reg [2:0] reg_flag;
reg [39:0] reg_str;
reg [31:0] signextend_as_unsigned;




always @ (instruction, regA, regB)

begin
    opcode = instruction[31:26];
    immediate = instruction[15:0];
    reg_flag = 3'b000;  // reset the flag before each operation
    
    rs = regA;
    rt = regB;


    //R type
    if (opcode == 6'b000000)          
    begin
        func = instruction[5:0];
        shift_amount = instruction[10:6];
        
        case(func)
        6'b100000:  // add
        begin
            reg_str = "add";
            reg_result33 = rs + rt;
            reg_result = rs + rt;
            if (reg_result33[32] == 1)  //overflow
                reg_flag = 3'b001;
        end

        6'b100001: // addu
        begin
            reg_str = "addu";
            reg_result = rs + rt;
        end
                    
        6'b100100: // and
        begin
            reg_str = "and";
            reg_result = rs & rt;
        end

        6'b100010: // sub
        begin
            reg_str = "sub";
            reg_result = rs - rt;
            reg_result33 = rs + rt;
            if (reg_result33[32] == 1)  // overflow
                reg_flag = 3'b001;

        end
                    
        6'b100011: // subu
        begin
            reg_str = "subu";
            reg_result = rs - rt;
        end
                    
        6'b100111: // nor
        begin
            reg_str = "nor";
            reg_result = ~(rs | rt);
        end

        6'b100101: // or
        begin
            reg_str = "or";
            reg_result = rs | rt;
        end


        6'b100110: // xor
        begin
            reg_str = "xor";
            reg_result = rs ^ rt;
        end
                    
        6'b101010: // slt
        begin
            reg_str = "slt";
            reg_result =  rs < rt;               // MODIFICATION: reg_result is (RS < RT), i.e., 0 or 1
            if (rs < rt) 
                reg_flag = 3'b010;
        end

        6'b101011: // sltu
        begin
            reg_str = "sltu";
            reg_result = $unsigned(rs) - $unsigned(rt);
            if ($unsigned(rs) < $unsigned(rt)) 
                reg_flag = 3'b010;
        end

        6'b000000: // sll
        begin
            reg_str = "sll";
            reg_result = rt << shift_amount;
        end

        6'b000100: // sllv
        begin
            reg_str = "sllv";
            reg_shift = rs;      // select the lower 5 bits
            reg_result = rt << reg_shift;
        end

        6'b000010: // srl
        begin
            reg_str = "srl";
            reg_result = rt >> shift_amount;
        end

        6'b000110: // srlv
        begin
            reg_str = "srlv";
            reg_shift = rs;      // select the lower 5 bits
            reg_result = rt >> reg_shift;
        end

        6'b000011: // sra
        begin
            reg_str = "sra";
            reg_result = rt >>> shift_amount;
        end

        6'b000111: // srav
        begin
            reg_str = "srav";
            reg_shift = rs;     // select the lower 5 bits
            reg_result = rt >>> reg_shift;
        end
        endcase

    end

    // I type
    else begin

        signextend_as_unsigned = {{16{immediate[15]}}, immediate};

    case(opcode)
    
    6'b001000:    // addi
    begin
        reg_str = "addi";
        reg_result = rs + immediate; 
        reg_result33 = rs + rt;
        if (reg_result33[32] == 1)      // overflow
            reg_flag = 3'b001;
    end

    6'b001001: // addiu
    begin
        reg_str = "addiu";
        reg_result = rs + immediate;
    end

    6'b001100: // andi
    begin
        reg_str = "andi";
        reg_result = rs & $unsigned(immediate);
    end

    6'b001101: // ori
    begin
        reg_str = "ori";
        reg_result = rs | $unsigned(immediate);
    end

    6'b001110: // xori
    begin
        reg_str = "xori";
        reg_result = rs ^ $unsigned(immediate);
    end

    6'b000100: // beq
    begin
        reg_str = "beq";
        reg_result = rs - rt;
        if (rs == rt) reg_flag = 3'b100;
    end

    6'b000101: // bne
    begin
        reg_str = "bne";
        reg_result = rs - rt;
        if (rs != rt)                          // MODIFICATION: ZERO FLAG IS 1 WHEN NOT EQUAL
            reg_flag = 3'b100;
    end

    6'b001010: // slti
    begin
        reg_str = "slti";
        reg_result = rs - immediate;
        if (rs < immediate) reg_flag = 3'b010;
    end

    6'b001011: // sltiu
    begin
        reg_str = "sltiu";
        reg_result = rs - signextend_as_unsigned;
        if ($unsigned(rs) < signextend_as_unsigned) 
            reg_flag = 3'b010;

    end

    6'b100011: // lw
    begin
        reg_str = "lw";
        reg_result = rs + immediate;

    end

    6'b101011: // sw
    begin
        reg_str = "sw";
        reg_result = rs + immediate;

    end
    
    endcase
    end

result = reg_result[31:0];
zeroflag = reg_flag[2];

end   


endmodule





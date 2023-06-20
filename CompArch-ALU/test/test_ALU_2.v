`timescale 1ns/1ps

module test();
reg  [31:0] instruction;
reg  [31:0] regA;
reg  [31:0] regB;
wire [31:0] result;
wire [2:0] flags;

  initial begin
    $dumpfile("test.vcd");
    $dumpvars;
  end

  initial begin
	$display("\n                                   initialize");
	$monitor("instruction = 32'h%h, regA = 32'h%h, regB = 32'h%h, result = 32'h%h, flags = 3'b%b",instruction,regA,regB,result,flags);
        instruction = 32'b0;
        regA = 32'b0; 
        regB = 32'b0;
        #500
	$display("\n                                   for R Type");
	$display("\n                                   format is");
	$display("\n                      instruction:regA:regB:result:flags");
	$display("\n                                add overflow test");
        instruction = 32'h00200020;   // testing for add overflow
        regA = 32'h7FFFFFFF;
        regB = 32'h00000006;
        #100
	$display("\n                                add normal test");
	instruction = 32'h00010020;   // testing for add normal
	regA = 32'h00000001;
	regB = 32'h00000002;
	#100
	$display("\n                                sub overflow test");
        instruction = 32'h00010022;   // testing for sub overflow
        regA = 32'h7FFFFFFF;
        regB = 32'hFFFFFFFF;
	#100
	$display("\n                                sub normal test");
	instruction = 32'h00010022;   // testing for sub normal
	regA = 32'h00000004;
	regB = 32'h00000003;
	#100
	$display("\n                                  addu test");
	instruction = 32'h00200021;   // testing for addu 
	regA = 32'h7FFFFFFF;
	regB = 32'h00000006;
	#100 
	$display("\n                                  subu test");
	instruction = 32'h00010023;   // testing for subu
	regA = 32'h7FFFFFFF;
	regB = 32'hFFFFFFFF;
	#100
	$display("\n                                  and test");
	instruction = 32'h00010024;   // testing for and
	regA = 32'hFFFFFFFF;
	regB = 32'h00000001;
	#100
	$display("\n                                   or test");
	instruction = 32'h00200025;   // testing for or
	regA = 32'h00000000;
	regB = 32'h00000001;
	#100
	$display("\n                                  xor test");
	instruction = 32'h00200026;  // testing for xor
	regA = 32'hFFFFFFFF;
	regB = 32'h0000000F;
	#100
	$display("\n                                  nor test");
	instruction = 32'h00200027;  //testing for nor
	regA = 32'h00000000;
	regB = 32'hFFFFFFFF;
	#100
	$display("\n                             slt test(negative)");
	instruction = 32'h0001002A;  // testing for slt
	regA = 32'h00000000;
	regB = 32'h00000001;
	#100
	$display("\n                          slt test(non-negative)");
	instruction = 32'h0001002A;
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                           sltu test(negative)");
	instruction = 32'h0001002B;  // testing for sltu
	regA = 32'h00000000;
	regB = 32'hFFFFFFFF;
	#100
	$display("\n                                sltu test");
	instruction = 32'h0001002B;  // testing for sltu
	regA = 32'hFFFFFFFF;
	regB = 32'hFFFFFFFE;
	#100
	$display("\n                                sll test");
	instruction = 32'h00010280;  // testing for sll (10)
	regA = 32'h00000000;
	regB = 32'h00000001;
	#100
	$display("\n                                srl test");
	instruction = 32'h00010282;  //testing for srl(10)
	regA = 32'h00000000;
	regB = 32'hF0000000;
	#100
        $display("\n                                sllv test");
	instruction = 32'h00010004; //testing for sllv
	regA = 32'h00000010;
	regB = 32'h00000001;
	#100
	$display("\n                                srlv test");
	instruction = 32'h00010006; // testing for srlv
	regA = 32'h00000004;
	regB = 32'h00000100;
	#100
	$display("\n                                srav test");
	instruction = 32'h00010007; // testing for srav
	regA = 32'h00000001;
	regB = 32'h00000100;
	#100
	$display("\n                                sra test");
	instruction = 32'h00010083; // testing for sra
	regA = 32'h00000000;
	regB = 32'h10000000;
	#100
	$display("\n                                         for I Type");
	$display("\n                                         format is");
	$display("\n                        instruction:regA:regB:immediate:result:flags");
	$monitor("instruction = 32'h%h, regA = 32'h%h, regB = 32'h%h, immediate = 16'b%b, result = 32'h%h, flags = 3'b%b",instruction,regA,regB,instruction[15:0],result,flags);
	$display("\n                                         addi test");
	instruction = 32'h20200001;  // testing for addi
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                                     addi test(overflow)");
	instruction = 32'h20200001;  // testing for addi
	regA = 32'h7FFFFFFF;
	regB = 32'h00000000;
	#100
	$display("\n                                        andi test");
	instruction = 32'h30010001;  // testing for andi
	regA = 32'hFFFFFFFF;
	regB = 32'h00000000;
	#100
	$display("\n                                        addiu test");
	instruction = 32'h22018001;  //testing for addiu
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                                         ori test");
	instruction = 32'h32010001; //testing for ori
	regA = 32'h00000010;
	regB = 32'h00000000;
	#100
	$display("\n                                        xori test");
	instruction = 32'h38010001; //testing for xori
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                                   slti test(negative)");
	instruction = 32'h28010001; //testing for slti
	regA = 32'h00000000;
	regB = 32'h00000000;
	#100
	$display("\n                                        slti test");
	instruction = 32'h28010000; //testing for slti
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                                     beq test(zero)");
	instruction = 32'h10010001; //testing for beq
	regA = 32'h00000001;
	regB = 32'h00000001;
	#100
	$display("\n                                   beq test(non-zero)");
	instruction = 32'h10010001; //testing for beq
	regA = 32'h00000000;
	regB = 32'h00000001;
	#100
	$display("\n                                   bne test(non-zero)");
	instruction = 32'h14010001; //testing for bne
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                                     bne test(zero)");
	instruction = 32'h14010001; //testing for bne
	regA = 32'h00000001;
	regB = 32'h00000001;
	#100
	$display("\n                                  sltiu test(negative)");
	instruction = 32'h2C010010; //testing for sltiu
	regA = 32'h00000001;
	regB = 32'h00000000;
	#100
	$display("\n                                sltiu test(non-negative)");
	instruction = 32'h2C010010; //testing for sltiu
	regA = 32'h00000100;
	regB = 32'h00000000;
	#100
	$display("\n                                        lw test");
	instruction = 32'h8C01700f; //testing for lw
	regA = 32'h00000100;
	regB = 32'h00000000;
	#100
	$display("\n                                        sw test");
	instruction = 32'hAC01700f; //testing for sw
	regA = 32'h00000100;
	regB = 32'h00000000;
	#100
    $stop;
  end

alu  u_add(
    .instruction( instruction ),
    .regA( regA ),
    .regB( regB ),
    .result( result ),
    .flags(flags)
);

endmodule


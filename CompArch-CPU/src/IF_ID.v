// IF/ID stage register
// Write instruction to register (sync falling edge)
// output InsOut, PC_plus4_out;


module IF_ID(InsIn, PC_4_In, InsOut, PC_4_out, clk, reset);

output reg [31:0] InsOut, PC_4_out;
input [31:0] InsIn, PC_4_In;
input clk, reset;


initial begin
    InsOut <= 0;                                        
    PC_4_out <= 0;   
end

always@(posedge clk)
    begin

      if(reset) begin
        InsOut <= 32'b0;
        PC_4_out <= 32'b0;
      end

      else begin
        	InsOut <= InsIn;
          PC_4_out <= PC_4_In;
      end

    end


endmodule

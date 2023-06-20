`timescale 100fs/100fs
// generate clock signals

module clock(CLK, EN);
input EN;
output reg CLK;

parameter Cycle = 2000;

initial begin
    CLK=0;
end

always begin
    if(EN==1)
    begin
        #(Cycle/2);
        CLK=~CLK;
    end
end

endmodule
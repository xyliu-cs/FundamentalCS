To test the functionality of the designed ALU module and view the results, run the following command:

```bash
make
iverilog -o out test_alu.v alu.v
vvp out
```

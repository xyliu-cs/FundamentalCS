.data
num: .word 13

.text
lui $t2, 80 # these lines give demo on how lw/lb/lh and sw/sb/sh works
ori $s0, $t2, 0 
add $s1, $s0, $zero
lw $s0, 0($s0) 
sw $s0, 4($s1) 

addi $v0, $zero, 10
syscall

.data
num: .word 13

.text
addi $v0, $zero, 5
syscall
add $t0, $zero, $v0
addi $v0, $zero, 5
syscall
add $t1, $zero, $v0
add $a0, $t1, $t0
addi $v0, $zero, 1
syscall

addi $v0, $zero, 10
syscall


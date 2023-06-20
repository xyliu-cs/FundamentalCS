

import ctypes
import re
import os
import sys

# filename declaration
asmfile = sys.argv[1]
mcfile = sys.argv[2]
checkfile = sys.argv[3]
infile = sys.argv[4]
outfile = sys.argv[5]

# reading/writing line pointer
inline = 0
outline = 0

# pointers 
static_data_pointer = 0x500000        # Start at 1 MB above
text_pointer = 0x400000 

# allocate memory
memory_size = 10 * 1024 * 1024              # 6 MB
memory = ['00000000' for i in range(0, memory_size)] 

reg_table = {

    "00000": "zero",  "00001": "at",  "00010": "v0", "00011": "v1",
    "00100": "a0",    "00101": "a1",  "00110": "a2", "00111": "a3",
    "01000": "t0",    "01001": "t1",  "01010": "t2", "01011": "t3",
    "01100": "t4",    "01101": "t5",  "01110": "t6", "01111": "t7",
    "11000": "t8",    "11001": "t9",  "10000": "s0", "10001": "s1",
    "10010": "s2",    "10011": "s3",  "10100": "s4", "10101": "s5",
    "10110": "s6",    "10111": "s7",  "11010": "k0", "11011": "k1",
    "11100": "gp",    "11101": "sp",  "11110": "fp", "11111": "ra" 
}

# register values are integers
reg_mem = {

    "zero": 0, "at": 0, "v0": 0, "v1": 0,
    "a0": 0,   "a1": 0, "a2": 0, "a3": 0,
    "t0": 0,   "t1": 0, "t2": 0, "t3": 0,
    "t4": 0,   "t5": 0, "t6": 0, "t7": 0,
    "t8": 0,   "t9": 0, "s0": 0, "s1": 0,
    "s2": 0,   "s3": 0, "s4": 0, "s5": 0,
    "s6": 0,   "s7": 0, "k0": 0, "k1": 0,
    "gp": 0x508000,   "sp": 0xA00000 -1, "fp": 0xA00000 -1, "ra": 0,
    "pc": 0x400000 ,  "hi": 0,   "lo": 0

}

# Supplementary function
def unsigned_right_shift(n, i):
    if n < 0:
        n = ctypes.c_uint32(n).value
    if i < 0:
        return -int_overflow(n << abs(i))
    return int_overflow(n >> i)

def int_overflow(val):
    maxint = 2147483647
    if not -maxint-1 <= val <= maxint:
        print('Overflow Error!')
        val = (val + (maxint + 1)) % (2 * (maxint + 1)) - maxint - 1
    return val


def str_sign_extend(stri):
    if stri[0] == '1':
        num = int(stri, 2) - (2 ** (len(stri)))
    elif stri[0] == '0':
        num = int(stri, 2)
    else:
        print("str_sign_extend error! String is: ", stri)
        return
    return num

def store_ascii_string(mem_addr, stri):
    global memory
    raw_length = len(stri)
    for j in range(0, len(stri)):
        if stri[j] == '\\':
            continue
        elif stri[j-1] == '\\':
            if stri[j] == 'n':
                raw_length -= 1
                memory[mem_addr] = bin(ord('\n'))[2:].zfill(8)
                mem_addr += 1
                continue
            else:
                print("Escape symbol error!", stri[j])
                return
        else:
            memory[mem_addr] = bin(ord(stri[j]))[2:].zfill(8)
            mem_addr += 1
    quotient = raw_length % 4
    if quotient != 0:
        for i in range(0, 4-quotient):
            memory[mem_addr+i] = 0






# R type functions

def ADD(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    try:
        reg_mem[rd_reg] = reg_mem[rs_reg] + reg_mem[rt_reg]
    except:
        print("error occur!!!")
        print('rs = ', rs_reg, 'value = ', reg_mem[rs_reg])
        print('rt= ', rt_reg, 'value= ',reg_mem[rt_reg])
        print('Current PC: ', reg_mem['pc'] )
        sys.exit()
    reg_mem["pc"] += 4


def ADDU(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    sval = reg_mem[rs_reg]
    tval = reg_mem[rt_reg]

    if sval < 0:
        sval += 2 ** 32
    if tval < 0:
        tval += 2 ** 32
    
    reg_mem[rd_reg] = int_overflow(sval + tval)
    reg_mem["pc"] += 4
    


def DIV(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    reg_mem["hi"] = (reg_mem[rs_reg] & 0xffffffff) % (reg_mem[rt_reg] & 0xffffffff)
    reg_mem["lo"] = int((reg_mem[rs_reg] & 0xffffffff)/ (reg_mem[rt_reg] & 0xffffffff))
    reg_mem["pc"] += 4

def DIVU(rs, rt, rd, sa):
    DIV(rs, rt, rd, sa)

def MULT(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    output = (reg_mem[rs_reg] & 0xffffffff) * (reg_mem[rt_reg] & 0xffffffff)
    reg_mem[rd_reg] = int_overflow(output)

    reg_mem["pc"] += 4

def MULTU(rs, rt, rd, sa):
    MULT(rs, rt, rd, sa)

def SUB(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rs_reg] - reg_mem[rt_reg]
    reg_mem["pc"] += 4

def SUBU(rs, rt, rd, sa):
    SUB(rs, rt, rd, sa)


def AND(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rs_reg] & reg_mem[rt_reg]
    reg_mem["pc"] += 4

def OR(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rs_reg] | reg_mem[rt_reg]
    reg_mem["pc"] += 4

def NOR(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = ~(reg_mem[rs_reg] ^ reg_mem[rt_reg])
    reg_mem["pc"] += 4

def XOR(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rs_reg] ^ reg_mem[rt_reg]
    reg_mem["pc"] += 4

def MFHI(rs, rt, rd, sa):
    global reg_mem
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem['hi']
    reg_mem["pc"] += 4

def MFLO(rs, rt, rd, sa):
    global reg_mem
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem['lo']
    reg_mem["pc"] += 4

def MTHI(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    reg_mem['hi'] = reg_mem[rs_reg]
    reg_mem["pc"] += 4

def MTLO(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    reg_mem['lo'] = reg_mem[rs_reg]
    reg_mem["pc"] += 4

def SLL(rs, rt, rd, sa):
    global reg_mem
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rt_reg] << int(sa, 2)
    reg_mem["pc"] += 4


def SLLV(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rt_reg] << int(bin(reg_mem[rs_reg])[:7], 0)
    reg_mem["pc"] += 4


def SLT(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    if reg_mem[rs_reg] < reg_mem[rt_reg]:
        reg_mem[rd_reg] == 1
    else:
        reg_mem[rd_reg] == 0
    reg_mem["pc"] += 4


def SLTU(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    sval = reg_mem[rs_reg]
    tval = reg_mem[rt_reg]
    reg_mem["pc"] += 4

    if sval < 0:
        sval = reg_mem[rs_reg] + 2 ** 32
    if tval < 0:
        tval = reg_mem[rt_reg] + 2 ** 32
    
    if sval < tval:
        reg_mem[rd_reg] = 1
    elif sval >= tval:
        reg_mem[rd_reg] = 0

def SRA(rs, rt, rd, sa):
    global reg_mem
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rt_reg] >> int(sa, 2)
    reg_mem["pc"] += 4

def SRAV(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = reg_mem[rt_reg] >> int(bin(reg_mem[rs_reg])[:7], 0)
    reg_mem["pc"] += 4

def SRL(rs, rt, rd, sa):
    global reg_mem
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]

    reg_mem[rd_reg] = unsigned_right_shift(reg_mem[rt_reg], int(sa, 2))
    
    reg_mem["pc"] += 4
        
def SRLV(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    rd_reg = reg_table[rd]
    reg_mem[rd_reg] = unsigned_right_shift(reg_mem[rt_reg], int(bin(reg_mem[rs_reg])[:7], 0))
    reg_mem["pc"] += 4

# R type jump

def JR(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    reg_mem["pc"] = reg_mem[rs_reg] 

def JALR(rs, rt, rd, sa):
    global reg_mem
    rs_reg = reg_table[rs]
    rd_reg = reg_table[rd]   
    reg_mem[rd_reg] = reg_mem["pc"] + 4 
    reg_mem["pc"] = reg_mem[rs_reg] 


# SYSTEM CALL
def SYSCALL(rs, rt, rd, sa):
    global reg_mem
    global inline
    global outline
    global memory


    # print_int()
    if reg_mem["v0"] == 1:
        # print('SYS_PRINT_INT: ', reg_mem["t2"])
        with open(outfile,'a+') as f:
            f.write(str(reg_mem["a0"]))
    
    # print_str()
    elif reg_mem["v0"] == 4:
        # print('SYS_PRINT_STR: ')
        with open(outfile,'a+') as f:
            addr = reg_mem["a0"]
            while memory[addr] != 0 and memory[addr] != '00000000':
                item = chr(int(memory[addr],2)) 
                # print(item)
                f.write(item)
                print(item, end='')
                addr += 1
            
    # read_int()
    elif reg_mem["v0"] == 5:
        with open(infile, 'r') as f:
            lines = f.readlines()
            reg_mem["v0"] = int(lines[inline])
            inline += 1

    # read_str()
    elif reg_mem["v0"] == 8:

        with open(infile, 'r') as f:
            lines = f.readlines()
            strline = lines[inline][:reg_mem["a1"]+1]
            store_ascii_string(reg_mem["a0"], strline)
            inline += 1
   
    # sys_sbrk
    elif reg_mem["v0"] == 9:
        print('calling sbrk: not well defined')
        reg_mem["pc"] += 4
        return
    
    # exit
    elif reg_mem["v0"] == 10:
        print('\nExcution finished!')
        sys.exit()
    
    # print_char()
    elif reg_mem["v0"] == 11:
        with open(outfile,'a+') as f:
            f.write(chr(reg_mem["a0"]))
            print(chr(reg_mem["a0"]))
   
    # read_char()
    elif reg_mem["v0"] == 12:
        with open(infile, 'r') as f:
            lines = f.readlines()
            reg_mem["v0"] = ord(lines[inline][0])
            inline += 1
    # open()
    elif reg_mem["v0"] == 13:

        addr = reg_mem["a0"]
        filename = ''
        while memory[addr] != 0 and memory[addr] != '00000000':
            item = chr(int(memory[addr],2))
            filename += item 
            addr += 1

        reg_mem["a0"] = os.open(filename, os.O_RDWR|os.O_CREAT)
        

    # read()
    elif reg_mem["v0"] == 14:
        filebyte = os.read(reg_mem["a0"],reg_mem["a2"])

        for i in range(len(filebyte)):
            memory[reg_mem["a1"]+i] = bin(filebyte[i])[2:].zfill(8)
       
        memory[reg_mem["a1"]+reg_mem["a2"]] = '00000000' #'\0'
        reg_mem["a0"] = reg_mem["a2"]

    # write()
    elif reg_mem["v0"] == 15:
        content = bytes()
        for i in range(reg_mem["a2"]):
            ascii = chr(int(memory[reg_mem["a1"]+i], 2))
            content += ascii.encode()
        os.write(reg_mem["a0"], content)

    # sys_close()
    elif reg_mem["v0"] == 16:
        os.close(reg_mem["a0"])
    
    # sys_exit2()
    elif reg_mem["v0"] == 17:
        print('\nExcution finished!')
        sys.exit()
    else:
        print("Unsupported system call: ", reg_mem["v0"])
        return
   
    reg_mem["pc"] += 4







# I type functions


def ADDI(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    reg_mem[rt_reg] = reg_mem[rs_reg] + str_sign_extend(imme)
    reg_mem["pc"] += 4

    
def ADDIU(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    reg_mem[rt_reg] = int_overflow(reg_mem[rs_reg] + int(imme, 2))
    reg_mem["pc"] += 4  

def ANDI(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    reg_mem[rt_reg] = reg_mem[rs_reg] & int(imme, 2)
    reg_mem["pc"] += 4

  
def BEQ(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    if  reg_mem[rs_reg] == reg_mem[rt_reg]:
        reg_mem["pc"] += 4 + int(imme, 2) << 2
    else:
        reg_mem["pc"] += 4


def BGEZ(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    if  reg_mem[rs_reg] >= 0:
        reg_mem["pc"] += 4 + int(imme, 2) << 2
    else:
        reg_mem["pc"] += 4

def BGTZ(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    if  reg_mem[rs_reg] > 0:
        reg_mem["pc"] += 4 + int(imme, 2) << 2
    else:
        reg_mem["pc"] += 4

def BLEZ(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    if  reg_mem[rs_reg] <= 0:
        reg_mem["pc"] += 4 + int(imme, 2) << 2
    else:
        reg_mem["pc"] += 4

def BNE(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    if  reg_mem[rs_reg] != reg_mem[rt_reg]:
        reg_mem["pc"] += 4 + int(imme, 2) << 2
    else:
        reg_mem["pc"] += 4   

# negative -> negative

def LB(rs, rt, imme):
    global reg_mem
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme 
    try:
        int(addr) >= 0
    except:
        print("LB address error:", addr)
        return
    target = memory[addr]
    if target == None:
        print('Error: loading machine code')
        return
    else:
        try: 
            int(target, 2)
        except:
            print("LB Cant to int: ", target)
            sys.exit()
    reg_mem[rt_reg] = str_sign_extend(target)
    reg_mem["pc"] += 4

# # set 1111 1100 -> 00...0 1111 1100 equals to treat it as a postive number

def LBU(rs, rt, imme):
    global reg_mem
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme
    
    try:
        int(addr) >= 0
    except:
        print("LBU address error:", addr)
        return
    
    target = memory[addr]
    if target == None:
        print('Error: loading machine code')
        return
    else:
        try: 
            target = int(target, 2)
        except:
            print("LBU Cant to int: ", target)
            sys.exit()
    reg_mem[rt_reg] = target
    reg_mem["pc"] += 4


def LH(rs, rt, imme):
    global reg_mem
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme 
    try:
        int(addr) >= 0 and int(addr) % 2 == 0
    except:
        print("LH address error:", addr)
        return

    # Use string concatenation
    target = memory[addr+1] + memory[addr] 
    
    if target == None:
        print('Error: loading machine code')
        return
    else:
        try: 
            int(target, 2)
        except:
            print("LH Cant to int: ", target)
            sys.exit()
    reg_mem[rt_reg] = str_sign_extend(target)
    reg_mem["pc"] += 4

def LHU(rs, rt, imme):
    global reg_mem
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme
    try:
        int(addr) >= 0 and int(addr) % 2 == 0
    except:
        print("LHU address error:", addr)
        return

    # Use string concatenation
    target = memory[addr+1] + memory[addr] 
    
    if target == None:
        print('Error: loading machine code')
        return
    else:
        try: 
            target = int(target, 2)
        except:
            print("LHU Cant to int: ", target)
            sys.exit()
    reg_mem[rt_reg] = target
    reg_mem["pc"] += 4

def LW(rs, rt, imme):
    global reg_mem
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme
    try:
        int(addr) >= 0 and int(addr) % 4 == 0
    except:
        print("LW address error:", addr)
        return

    # Use string concatenation
    target = memory[addr+3] + memory[addr+2] + memory[addr+1] + memory[addr]
    
    if target == None:
        print('Error: loading machine code')
        return
    else:
        try: 
            int(target, 2)
        except:
            print("LW Cant to int: ", target)
            sys.exit()
    reg_mem[rt_reg] = int(target, 2)
    reg_mem["pc"] += 4

   
def LUI(rs, rt, imme):
    global reg_mem
    rt_reg = reg_table[rt]
    # make immedieate upper 16 digits
    num_stri = imme + '0000000000000000'
    num = int(num_stri, 2)
    reg_mem[rt_reg] = num
    reg_mem["pc"] += 4

def SB(rs, rt, imme):
    global reg_mem
    global memory
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme

    try:
        int(addr) >= 0
    except:
        print("SB address error:", addr)
        return

    num = int(reg_mem[rt_reg])
    num = bin(num & 0xffffffff)[2:].zfill(32)
    num_low = num[24:]
       
    # store in the form of string
    memory[addr] = num_low
    reg_mem["pc"] += 4

def SH(rs, rt, imme):
    global reg_mem
    global memory
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme 
    try:
        int(addr) >= 0 and int(addr) % 2 == 0
    except:
        print("SH address error:", addr)
        return
   
    num = int(reg_mem[rt_reg]) 
    num = bin(num & 0xffffffff)[2:].zfill(32)
    num_low = num[24:]
    num_hig = num[16:24]
       
    # store in the form of string
    memory[addr] = num_low
    memory[addr+1] = num_hig
    reg_mem["pc"] += 4  

def SW(rs, rt, imme):
    global reg_mem
    global memory
    rt_reg = reg_table[rt]
    rs_reg = reg_table[rs]
    imme = str_sign_extend(imme)
    addr = reg_mem[rs_reg] + imme 
    try:
        int(addr) >= 0 and int(addr) % 4 == 0
    except:
        print("SW address error:", addr)
        return
 
    num = int(reg_mem[rt_reg]) 
    num = bin(num & 0xffffffff)[2:].zfill(32)
    num_low = num[24:]
    num_low_mid = num[16:24]
    num_hih_mid = num[8:16]
    num_hih = num[:8]
       
    # store in the form of string
    memory[addr] = num_low
    memory[addr+1] = num_low_mid
    memory[addr+1] = num_hih_mid
    memory[addr+1] = num_hih
    reg_mem["pc"] += 4  

def ORI(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    reg_mem[rt_reg] = reg_mem[rs_reg] | int(imme, 2)
    reg_mem["pc"] += 4

def XORI(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    reg_mem[rt_reg] = reg_mem[rs_reg] ^ int(imme, 2)
    reg_mem["pc"] += 4

def SLTI(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    imme = str_sign_extend(imme)
    if imme > int(reg_mem[rs_reg]):
        reg_mem[rt_reg] = 1
    else:
        reg_mem[rt_reg] = 0
    reg_mem["pc"] += 4

def SLTIU(rs, rt, imme):
    global reg_mem
    rs_reg = reg_table[rs]
    rt_reg = reg_table[rt]
    imme = str_sign_extend(imme)
    if imme > (int(reg_mem[rs_reg]) & 0xffffffff):
        reg_mem[rt_reg] = 1
    else:
        reg_mem[rt_reg] = 0
    reg_mem["pc"] += 4

def LWL(rs, rt, imme):
    global reg_mem
    print("Not supported type: LWL")
    reg_mem["pc"] += 4

def LWR(rs, rt, imme):
    global reg_mem
    print("Not supported type: LWR")
    reg_mem["pc"] += 4

def SWR(rs, rt, imme):
    global reg_mem
    print("Not supported type: SWR")
    reg_mem["pc"] += 4

def SWL(rs, rt, imme):
    global reg_mem
    print("Not supported type: SWL")
    reg_mem["pc"] += 4

# J type functions

def J(target):
    global reg_mem
    temp_pc = reg_mem["pc"]
    temp_pc = bin(temp_pc+4)[2:].zfill(32)
    high_digit = temp_pc[:4]
    addr = int((high_digit + target + '00'), 2) 
    if addr < 0:
        print("Address < 0. Address: ", addr)
        return
    else:
        reg_mem["pc"] = addr
    
def JAL(target):
    global reg_mem
    temp_pc = reg_mem["pc"]
    temp_pc_bin = bin(temp_pc+4)[2:].zfill(32)
    high_digit = temp_pc_bin[:4]
    addr = int((high_digit + target + '00'), 2) 
    if addr < 0:
        print("Address < 0. Address: ", addr)
        return
    else:
        reg_mem["pc"] = addr
    
    reg_mem["ra"] = temp_pc + 4








RFunctions={

    '100000': ADD, '100001': ADDU, '100100': AND, '011010': DIV,
    '011011': DIVU, '001001': JALR, '001000': JR, '010000': MFHI,
    '010010': MFLO, '010001': MTHI, '010011': MTLO, '011000': MULT,
    '011001': MULTU, '100111': NOR, '100101': OR, '000000': SLL,    
    '000100': SLLV, '101010': SLT, '101011': SLTU, '000011': SRA,
    '000111': SRAV, '000010': SRL, '000110': SRLV, '100010': SUB,
    '100011': SUBU, '001100': SYSCALL, '100110': XOR
       
}

IFunctions={

    '001000': ADDI,  '001001': ADDIU, '001100': ANDI, '000100': BEQ,
    '000001': BGEZ,  '000111': BGTZ,  '000110': BLEZ,  #{0, BLTZ}, modified here 
    '000101': BNE,   '100000': LB,    '100100': LBU, '100001': LH,
    '100101': LHU,   '001111': LUI,   '100011': LW,  '001101': ORI,
    '101000': SB,    '001010': SLTI,  '001011': SLTIU,'101001': SH,
    '101011': SW,    '001110': XORI,  '100010': LWL,  '100110': LWR,
    '101010': SWL,   '101110': SWR
}


JFunctions={

    '000010': J, '000011': JAL
}















def store_static_data(file):

    global static_data_pointer
    global memory

    with open(file, "r") as f:

        codelines = f.readlines()

        data_flag = False

        for i in codelines:
            i = i.strip()

            if i == '' or i[0] == '#':
                continue
            if i.startswith(".data"):
                data_flag = True
                continue
            if i.startswith(".text"):
                data_flag = False
                break
            
            if data_flag:

            # iterating over the varible expression by chars
                if i.find('#') != -1:
                    comment = i.find('#')
                    i = i[:comment].strip()

                if i.find(':') != -1:
                    start = i.find(':')
                    i = i[start+1:].strip()


                if i.startswith(".asciiz"):
                    try:
                        quote1 = i.find("\"")
                        i_sechalf = i[quote1+1:]
                        quote2 = i_sechalf.find("\"")
                        i_string = i_sechalf[:quote2]

                    except:
                        print("No matching quotes, check input !")
                        return
                    
                    raw_length = len(i_string)
                    # tranfer char to their decimal ASCII code
                    for j in range(0, len(i_string)):
                        # wait for the n to store '\n' ascii
                        if i_string[j] == '\\':
                            continue
                        elif i_string[j-1] == '\\':
                            if i_string[j] == 'n':
                                raw_length -= 1
                                memory[static_data_pointer] = bin(ord('\n'))[2:].zfill(8)
                                static_data_pointer += 1
                            else:
                                print("Escape symbol error!", i_string[j])
                                return
                        else:
                            memory[static_data_pointer] = bin(ord(i_string[j]))[2:].zfill(8)
                            static_data_pointer += 1
                    # add '\0' at the end of the string, and make it word align
                    memory[static_data_pointer] = bin(ord('\0'))[2:].zfill(8)
                    quotient = (raw_length + 1) % 4
                    if quotient != 0:
                        static_data_pointer += (1 + 4-quotient)
                    else:
                        static_data_pointer += 1
         
                    
                elif i.startswith(".ascii"):
                    try:
                        quote1 = i.find("\"")
                        i_sechalf = i[quote1+1:]
                        quote2 = i_sechalf.find("\"")
                        i_string = i_sechalf[:quote2]

                    except:
                        print("No matching quotes, check input !")
                        return
                    
                    raw_length = len(i_string)
                 
                    # put char into memory in the form of str
                    for j in range(0, len(i_string)):
                        if i_string[j] == '\\':
                            continue
                        elif i_string[j-1] == '\\':
                            if i_string[j] == 'n':
                                raw_length -= 1
                                memory[static_data_pointer] = bin(ord('\n'))[2:].zfill(8)
                                static_data_pointer += 1
                            else:
                                print("Escape symbol error!", i_string[j])
                                return
                        else:
                            memory[static_data_pointer] = bin(ord(i_string[j]))[2:].zfill(8)
                            static_data_pointer += 1
                    quotient = raw_length % 4
                    if quotient != 0:
                        static_data_pointer += 4 - quotient
    
               
                elif i.startswith(".word"):
                    result = re.findall(r'[\d\|-]+',i)
                    # print(result)
                    for number in result:
                        num_to_store =  bin(int(number) & 0xffffffff)[2:].zfill(32)
                        memory[static_data_pointer] = num_to_store[24:]
                        memory[static_data_pointer+1] = num_to_store[16:24]
                        memory[static_data_pointer+2] = num_to_store[8:16]
                        memory[static_data_pointer+3] = num_to_store[0:8]
                    
                        static_data_pointer += 4
                
                elif i.startswith(".byte"):
                    result = re.findall(r'[\d\|-]+',i)                   
                    # print(result)
                    quotient = len(result) % 4
                    for number in result:
                        memory[static_data_pointer] = bin(int(number) & 0xff)[2:].zfill(8)                     
                        static_data_pointer += 1
                    if quotient != 0:
                        static_data_pointer += 4 - quotient


                elif i.startswith(".half"):
                    result = re.findall(r'[\d\|-]+',i)
                    # print(result)
                    quotient = (len(result)*2) % 4
                    for number in result:
                        num_to_store = bin(int(number) & 0xffff)[2:].zfill(16)
                        memory[static_data_pointer] = num_to_store[8:16]
                        memory[static_data_pointer+1] = num_to_store[0:8]
                        static_data_pointer += 2
                    if quotient != 0:
                        static_data_pointer += 4 - quotient

                else:
                    print("Error")
                    return
        f.close()
   

def read_machinecode(file):

    global memory
    global text_pointer
    
    
    with open(file, 'r') as f:
        ins_lines = f.readlines()
        for instruction in ins_lines:
            if instruction != '':
                buffer = instruction[:-1]
                memory[text_pointer] = buffer[24:32]
                memory[text_pointer+1] = buffer[16:24]
                memory[text_pointer+2] = buffer[8:16]
                memory[text_pointer+3] = buffer[:8]
                text_pointer += 4
        f.close()
            

def execute(checklist):
    global reg_mem
 
    #execute the machine code
    execute_time = 0
    while (True):
        if reg_mem["pc"] >= 0x500000:
            print("Program finished")
            break
        
        else:
            point = reg_mem["pc"]
            code_string = memory[point+3]+ memory[point+2] + memory[point+1] + memory[point]

            if execute_time in checklist:
                reg_bin = 'register_' + str(execute_time) + '.bin'
                mem_bin = 'memory_' + str(execute_time) + '.bin'
                with open(reg_bin, 'w') as f1:
                    for i in reg_mem.values():
                        val = bin(i)[2:].zfill(32)
                        f1.write(val+'\n')
                    f1.close()

                with open(mem_bin, 'w') as f2:
                    for i in range(0x400000,0xA00000):
                        f2.write(memory[i]+'\n')
                    f2.close()                


                    
            # R type
            if code_string.startswith('000000'):
                execute_time += 1
                rs = code_string[6:11]
                rt = code_string[11:16]
                rd = code_string[16:21]
                sa = code_string[21:26]
                func = code_string[26:]
                RFunctions[func](rs, rt, rd, sa)
            # J type
            elif code_string.startswith('00001'):
                execute_time += 1
                op = code_string[:6]
                target = code_string[6:]
                JFunctions[op](target)
            # I type
            else:
                execute_time += 1
                op = code_string[:6]
                rs = code_string[6:11]
                rt = code_string[11:16]
                imme = code_string[16:]
                IFunctions[op](rs, rt, imme)
            
            

                    

def main():

    checlist = []
    with open(checkfile, 'r') as f:
        print("open checkfile sucessfully")
        numlist = f.read().splitlines()
        for i in numlist:
            checlist.append(int(i))
        print('Check point list: ', checlist)
        f.close()            

    print('\n')
    store_static_data(asmfile)
    print("static data stored: ")
    print(memory[0x500000:static_data_pointer])
    print('\n')
    read_machinecode(mcfile)
    print('machine code loaded: ')
    print(memory[0x400000:text_pointer])
    print('\nexecution starts:')
    execute(checlist)
    


main()
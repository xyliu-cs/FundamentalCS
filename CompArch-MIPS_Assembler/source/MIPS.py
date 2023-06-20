
from glob import glob
from operator import truediv
import re




# global variable declarations
program_counter = 4194300
label_table = {}
linecode = []
machine_code = []

# register code
reg_table = {

    "zero": "00000",
    "at": "00001",
    "v0": "00010",
    "v1": "00011",
    "a0": "00100",
    "a1": "00101",
    "a2": "00110",
    "a3": "00111",
    "t0": "01000",
    "t1": "01001",
    "t2": "01010",
    "t3": "01011",
    "t4": "01100",
    "t5": "01101",
    "t6": "01110",
    "t7": "01111",
    "t8": "11000",
    "t9": "11001",
    "s0": "10000",
    "s1": "10001",
    "s2": "10010",
    "s3": "10011",
    "s4": "10100",
    "s5": "10101",
    "s6": "10110",
    "s7": "10111",
    "k0": "11010",
    "k1": "11011",
    "gp": "11100",
    "sp": "11101",
    "fp": "11110",
    "ra": "11111"
}

instruction_map = {
    
    # R type instructions
    # format: "instruction : [type, function_code, operands_tuple]

    "add": ["R", "100000", ('rd', 'rs', 'rt')],
    "addu": ["R", "100001", ('rd', 'rs', 'rt')],
    "and": ["R", "100100", ('rd', 'rs', 'rt')],
    "div": ["R", "011010", ('rs', 'rt')],
    "divu": ["R", "011011", ('rs', 'rt')],
    "jalr": ["R", "001001", ('rd', 'rs')],
    "jr": ["R", "001000", ('rs',)],
    "mfhi": ["R", "010000", ('rd',)],
    "mflo": ["R", "010010", ('rd',)],
    "mthi": ["R", "010001", ('rs',)],
    "mtlo": ["R", "010011", ('rs',)],
    "mult": ["R", "011000", ('rs', 'rt')],
    "multu": ["R", "011001", ('rs', 'rt')],
    "nor": ["R", "100111", ('rd', 'rs', 'rt')],
    "or": ["R", "100101", ('rd', 'rs', 'rt')],
    "sll": ["R", "000000", ('rd', 'rt', 'sa')],
    "sllv": ["R", "000100", ('rd', 'rt', 'rs')],
    "slt": ["R", "101010", ('rd', 'rs', 'rt')],
    "sltu": ["R", "101011", ('rd', 'rs', 'rt')],
    "sra": ["R", "000011", ('rd', 'rt', 'sa')],
    "srav": ["R", "000111", ('rd', 'rt', 'rs')],
    "srl": ["R", "000010", ('rd', 'rt', 'sa')],
    "srlv": ["R","000110", ('rd', 'rt', 'rs')],
    "sub": ["R", "100010", ('rd', 'rs', 'rt')],
    "subu": ["R", "100011", ('rd', 'rs', 'rt')],
    "syscall": ["R", "001100"],
    "xor": ["R", "100110", ('rd', 'rs', 'rt')],
    
    # I type instructions

    "addi": ["I", "001000", ('rt', 'rs', 'i')],
    "addiu": ["I", "001001", ('rt', 'rs', 'i')],
    "andi": ["I", "001100", ('rt', 'rs', 'i')],
    "beq": ["I", "000100", ('rs', 'rt', 'label')],
    "bgez": ["I", "000001", ('rs', 'label')],  # rt = 00001
    "bgtz": ["I", "001000", ('rs', 'label')],  # rt = 00000
    "blez": ["I", "000110", ('rs', 'label')],  # rt = 00000
    "bltz": ["I", "000001", ('rs', 'label')],  # rt = 00000
    "bne": ["I", "000101", ('rs', 'rt', 'label')],
    "lb": ["I", "100000", ('rt', 'i', 'rs')],
    "lbu": ["I", "100100", ('rt', 'i', 'rs')],
    "lh": ["I", "100001", ('rt', 'i', 'rs')],
    "lhu": ["I", "100101", ('rt', 'i', 'rs')],
    "lui": ["I", "001111", ('rt', 'i')],
    "lw": ["I", "100011", ('rt', 'i', 'rs')],
    "ori": ["I", "001101", ('rt', 'rs', 'i')],
    "sb": ["I", "101000", ('rt', 'i', 'rs')],
    "slti": ["I", "001010", ('rt', 'rs', 'i')],
    "sltiu": ["I", "001011", ('rt', 'rs', 'i')],  
    "sh": ["I", "101001", ('rt', 'i', 'rs')], 
    "sw": ["I", "101011", ('rt', 'i', 'rs')],
    "xori": ["I", "001110", ('rt', 'rs', 'i')],
    "lwl": ["I", "100010", ('rt', 'i', 'rs')],
    "lwr": ["I", "100110", ('rt', 'i', 'rs')],
    "swl": ["I", "101010", ('rt', 'i', 'rs')],
    "swr": ["I", "101110", ('rt', 'i', 'rs')],  

    # J type instructions

    "j": ["J", "000010"],
    "jal": ["J", "000011"]

}

def phase1 (filepath):
    global program_counter
    global label_table
    global linecode
    strlist = []
    flag_txt = False
    
    with open(filepath, 'r') as f:
        list1 = f.readlines()

        # iterating over lines in the file
        for i in list1:
            if i[0] == '#':
                continue
            if i.startswith('.text'):
                flag_txt = True
                continue
            elif i.startswith('.data'):
                flag_txt = False
                continue

        # iterating over chars in the lines
            if flag_txt:
                for k in i:
                    if k == '#':
                        break                    
                    elif k == '\n':                       
                        continue
                    else:
                        strlist.append(k)

                linecode.append(''.join(strlist).strip())
                strlist.clear() 
        
        # label mapping
        for i in range(len(linecode)):

            if linecode[i].find(':') == -1:
                program_counter += 4 
            
            else:
        # the instruction is below the label
                if linecode[i].strip().endswith(':'):    
                    label = linecode[i][:-1]
                                       
        # the label and the instruction are in the same line
                else:                   
                    strline = linecode[i]
                    label = strline[:strline.find(':')].strip()
                    
                label_table[label] = program_counter + 4
    print('\nLabel table is: \n', label_table, '\n')
    f.close()

    return linecode

def phase2 (code):
    global machine_code
    i = 1
    counter = 4194300
    for line in code:
        mcode = 0      
        linelist = re.findall(r"[\w\|-]+", line)
        # print('linelist is: ', linelist)
        # print('output code line', i)
       
        if linelist == []:
            continue
        elif linelist[0] not in instruction_map.keys():
            continue
        else:
            i += 1
            counter += 4 
            operation = linelist[0]
            info_list = instruction_map[operation]

            # R type instruction
            if info_list[0] == 'R':
                match_dic = {'rs': '00000', 'rt': '00000', 'rd': '00000', 'sa': '00000'}
                # print('operation: ', operation)
                # print('info_list: ', info_list)
                if operation == 'syscall':
                    mcode = '00000000000000000000000000001100'
                else:
                    r_tup = info_list[2]
                    # print('ins_tup: ', ins_tup)

                    # update the value in the match_dic, if not, stays the same
                    if len(r_tup) == 3 and len(linelist) == 4 and r_tup[2] != 'sa':  
                        match_dic[r_tup[0]] = reg_table[linelist[1]] 
                        match_dic[r_tup[1]] = reg_table[linelist[2]]
                        match_dic[r_tup[2]] = reg_table[linelist[3]]
                    # special case: transfer sa into binary number with 5 digits
                    # untested !!!!!!
                    elif len(r_tup) == 3 and len(linelist) == 4 and r_tup[2] == 'sa':  
                        match_dic[r_tup[0]] = reg_table[linelist[1]] 
                        match_dic[r_tup[1]] = reg_table[linelist[2]]
                        sa_bin = bin(int(linelist[3]))[2:].zfill(5)
                        match_dic[r_tup[2]] = sa_bin                        
                    elif len(r_tup) == 2 and len(linelist) == 3:
                        match_dic[r_tup[0]] = reg_table[linelist[1]] 
                        match_dic[r_tup[1]] = reg_table[linelist[2]]
                    elif len(r_tup) == 1 and len(linelist) == 2:
                        match_dic[r_tup[0]] = reg_table[linelist[1]]
                    else:
                        print('Index error, R type.')
                        print('the length of ins_tup is :', len(r_tup))
                        print('the length of linelist is :', len(linelist))
                        return
                    
                    mcode = '000000' + match_dic['rs'] + match_dic['rt'] + match_dic['rd'] + match_dic['sa'] + info_list[1]
            
            # I type instruction
            elif info_list[0] == 'I':

                i_dic = {'rs': '00000', 'rt': '00000', 'i': '0000000000000000'}
                i_tuple = info_list[2]
                
                if len(i_tuple) != (len(linelist) -1):
                    print('Index error, I type.')
                    print('the length of i_tuple is :', len(i_tuple))
                    print('the length of linelist is :', len(linelist))
                    return 

                for index in range(1, len(linelist)): #corresponding i_tuple index is less by 1
                    if i_tuple[index-1] == 'i':
                        if (linelist[index]) == 'zero':
                            continue
                        elif int(linelist[index]) >= 0:
                            i_dic["i"] = bin(int(linelist[index]))[2:].zfill(16)
                        else:
                            neg = int(linelist[index])
                            i_dic["i"] = bin(neg & 0xffff)[2:].zfill(16)
                    
                    elif i_tuple[index-1] == 'label':
                        tar_add = label_table[linelist[index]]
                        diff = tar_add - counter - 4 
                        # special instruction modification
                        if operation == 'bgez':
                            i_dic['rt'] = '00001'
                        
                        if diff >= 0:
                            immediate = bin(diff)[2:-2].zfill(16)                           
                        else:
                            # 1110 = 111111110
                            immediate = bin(diff & 0xffff)[2:-2].rjust(16, '1')
                            # print('immediate = ', immediate)                      
                        
                        i_dic['i'] = immediate
                        

                    else:
                        i_dic[i_tuple[index-1]] = reg_table[linelist[index]]

                    mcode = info_list[1] + i_dic["rs"] + i_dic["rt"] + i_dic["i"]                   


            # J type instruction
            elif info_list[0] == 'J':
                try: 
                    addr = bin(label_table[linelist[1]])[2:].zfill(32)     
                except:
                    print('Oops, something goes wrong in the address transformation! ')
                    return
                
                target_addr = addr[4:30]
                mcode = info_list[1] + target_addr
                        
            else:
                print('error, such type do not exist')
                return
            
            # print(mcode)
            machine_code.append(mcode)

            
    return machine_code      



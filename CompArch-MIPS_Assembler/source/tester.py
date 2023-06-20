import os
import MIPS

# ask the user to input files' names and check whether the files exist
test_file = input("Please enter the test file name: ")
while not os.path.exists(test_file):
    test_file = input("Your input is wrong, please enter the correct name: ")    

output_file = input("Please enter the output file name: ")
f = open(output_file, 'w')

expectedoutput_file = input("Please enter the expected output file name:")
while not os.path.exists(expectedoutput_file):
    expectedoutput_file=input("Your input is wrong, please enter the correct name: ")



# conduct phase2.py to do compiling
machine_code = MIPS.phase2(MIPS.phase1(test_file))
for line in machine_code:
    # print(line)
    f.write(line)
    f.write('\n')
f.close()


# read and store the content in the output file and expected outputfile seperately
with open(output_file,"r") as f1:
    output_lines=f1.readlines()

with open(expectedoutput_file,"r") as f2:
    expect_lines=f2.readlines()

# revise the content of the lists from two files for comparison
for i in range(0,len(output_lines)):
    line = output_lines[i]
    line = line.replace("\n","") #delete "\n"
    output_lines[i] = line
for i in range(0,len(expect_lines)):
    line = expect_lines[i]
    line = line.replace("\n","").replace(" ","").replace("\t","") #delete "\n","\t" and space
    expect_lines[i] = line

# compare two files' content and print the result
if output_lines == expect_lines:
    print("All Passed! Congratulations!")
else:
    print("You did something wrong!")
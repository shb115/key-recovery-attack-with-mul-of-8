import pickle
import numpy as np
import math

result = []

for i in range(522574, 522586):
    try:
        f = open("mul_of_8_key_recovery_%d.pickle"%(i), "rb")        
    except:
        continue

    while True:
        try:
            result.append(pickle.load(f))
        except EOFError:
            break
    f.close()

total=len(result)
#result=result[:100]
#print(len(result))
succ=0
fail=0
non=0

for i in result:
    if i==0: fail+=1
    elif i==1: succ+=1
    else: non+=1

print("total:",total)
print("succ :",succ)
print("fail :",fail)
print("non  :",non)
print("ratio:",succ/total)

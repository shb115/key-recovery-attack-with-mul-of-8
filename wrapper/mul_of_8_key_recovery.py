import CipherTools as CT
import random
import numpy as np
import math
import pickle
import os
import multiprocessing as mt

mk_len = 16
pt_len = 16

def worker(idx_tup):
    # SETTING
    pid = os.getpid()    
    mk = random.randint(0, 2 ** 128 - 1)
    mk_arry = CT.int_to_cbytes(mk, mk_len)
        
    # KEY_RECOVERY

    # First Diagonal
    pt = random.randint(0, 2 ** 128 - 1)
    pt_arry = CT.int_to_cbytes(pt, pt_len)
    res_succ = CT.MUL_OF_8_KEY_RECOVERY_FIRST_DIAGONAL(mk_arry, pt_arry, 5)
    if (res_succ == 0):
        f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
        pickle.dump(res_succ, f)
        f.close()
        return
    elif (res_succ == 2):
        pt = random.randint(0, 2 ** 128 - 1)
        pt_arry = CT.int_to_cbytes(pt, pt_len)
        res_succ = CT.MUL_OF_8_KEY_RECOVERY_FIRST_DIAGONAL(mk_arry, pt_arry, 5)
        if (res_succ == 0):
            f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
            pickle.dump(res_succ, f)
            f.close()
            return
        elif (res_succ == 2):
            pt = random.randint(0, 2 ** 128 - 1)
            pt_arry = CT.int_to_cbytes(pt, pt_len)
            res_succ = CT.MUL_OF_8_KEY_RECOVERY_FIRST_DIAGONAL(mk_arry, pt_arry, 5)
            if (res_succ == 0) or (res_succ == 2):
                f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
                pickle.dump(res_succ, f)
                f.close()
                return
    
    # Second Diagonal
    pt = random.randint(0, 2 ** 128 - 1)
    pt_arry = CT.int_to_cbytes(pt, pt_len)
    res_succ = CT.MUL_OF_8_KEY_RECOVERY_SECOND_DIAGONAL(mk_arry, pt_arry, 5)
    if (res_succ == 0):
        f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
        pickle.dump(res_succ, f)
        f.close()
        return
    elif (res_succ == 2):
        pt = random.randint(0, 2 ** 128 - 1)
        pt_arry = CT.int_to_cbytes(pt, pt_len)
        res_succ = CT.MUL_OF_8_KEY_RECOVERY_SECOND_DIAGONAL(mk_arry, pt_arry, 5)
        if (res_succ == 0):
            f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
            pickle.dump(res_succ, f)
            f.close()
            return
        elif (res_succ == 2):
            pt = random.randint(0, 2 ** 128 - 1)
            pt_arry = CT.int_to_cbytes(pt, pt_len)
            res_succ = CT.MUL_OF_8_KEY_RECOVERY_SECOND_DIAGONAL(mk_arry, pt_arry, 5)
            if (res_succ == 0) or (res_succ == 2):
                f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
                pickle.dump(res_succ, f)
                f.close()
                return
    
    # Third Diagonal
    pt = random.randint(0, 2 ** 128 - 1)
    pt_arry = CT.int_to_cbytes(pt, pt_len)
    res_succ = CT.MUL_OF_8_KEY_RECOVERY_THIRD_DIAGONAL(mk_arry, pt_arry, 5)
    if (res_succ == 0):
        f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
        pickle.dump(res_succ, f)
        f.close()
        return
    elif (res_succ == 2):
        pt = random.randint(0, 2 ** 128 - 1)
        pt_arry = CT.int_to_cbytes(pt, pt_len)
        res_succ = CT.MUL_OF_8_KEY_RECOVERY_THIRD_DIAGONAL(mk_arry, pt_arry, 5)
        if (res_succ == 0):
            f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
            pickle.dump(res_succ, f)
            f.close()
            return
        elif (res_succ == 2):
            pt = random.randint(0, 2 ** 128 - 1)
            pt_arry = CT.int_to_cbytes(pt, pt_len)
            res_succ = CT.MUL_OF_8_KEY_RECOVERY_THIRD_DIAGONAL(mk_arry, pt_arry, 5)
            if (res_succ == 0) or (res_succ == 2):
                f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
                pickle.dump(res_succ, f)
                f.close()
                return

    # SAVE
    f = open("mul_of_8_key_recovery_%s.pickle"%(pid), "ab")
    pickle.dump(res_succ, f)
    f.close()
    return

workload = [ [i] for i in range(1000)]

with mt.Pool(10) as p:
    p.map(worker, workload)
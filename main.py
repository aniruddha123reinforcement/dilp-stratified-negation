from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP

import argparse
import os
import random
from datetime import datetime
from math import sqrt
import pickle
import time
import shutil
import sys
import numpy as np
import itertools

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    sqr = int(sqrt(n)) + 1

    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True


def prime():
    div=[]
    for ni in range(2, 10):
      for no in range(2, 10):
        if no > 0 and (ni % no) == 0.0:
            div.append([ni, no])

    B = [Atom([Term(False, str(i)), Term(False, str(i))], 'equal')
         for i in range(2, 10)] +\
             [Atom([Term(False, str(i)), Term(False, str(j))], 'div')
         for i,j in div]

    primes = []
    not_primes = []
    for x in range(2, 10):
      if is_prime(x):
          primes.append(x)
      else:
          not_primes.append(x)

    P = [Atom([Term(False, str(i))], 'target')
         for i in primes]

    N = [Atom([Term(False, str(i))], 'target')
         for i in not_primes]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Atom([term_x_0, term_x_1], 'equal', strata=0),
           Atom([term_x_0, term_x_1], 'div', strata=0)]
    p_a = [Atom([term_x_0], 'pred', strata=1)]
    target = Atom([term_x_0], 'target', strata=2)

    p_a_rule = (Rule_Template(1, False), None)
    target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    rules = {p_a[0]: p_a_rule, target: target_rule} #  
    constants = [str(i) for i in range(2, 10)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 15)

    dilp = DILP(langage_frame, B, P, N, program_template,
                allow_negation=True, output_path='output')
    dilp.lr = 0.05
    dilp.reg_weight = 0.001 
    dilp.train(steps=1000)



prime()


'''Defines the ILP problem
'''

from src.ilp import Language_Frame, Program_Template
from src.core import Atom, Term
import numpy as np
from src.utils import check_neg_condition, negate_predicate
import tensorflow as tf

class ILP():

    def __init__(self, language_frame: Language_Frame, background: list, positive: list, negative: list, program_template: Program_Template, allow_negation=False):
        '''
        Arguments:
            language_frame {Language_Frame} -- language frame
            background {list} -- background assumptions
            positive {list} -- positive examples
            negative {list} -- negative examples
            program_template {Program_Template} -- program template
        '''
        self.language_frame = language_frame
        self.background = background
        self.positive = positive
        self.negative = negative
        self.program_template = program_template
        self.allow_negation = allow_negation

    def generate_ground_atoms(self):
        '''Generates the ground atoms from p_i,p_a,target and constants
        '''
        p = list(set(self.language_frame.p_e +
                     self.program_template.p_a + [self.language_frame.target]))
        constants = self.language_frame.constants

        # Build constant matrix
        constant_matrix = []
        for const1 in constants:
            for const2 in constants:
                term1 = Term(False, const1)
                term2 = Term(False, const2)
                constant_matrix.append([term1, term2])
        # Build ground atoms
        ground_atoms = []
        ground_atoms.append(Atom([], '⊥',strata=0))
        added_atoms = {}
        for pred in p:
            for term in constant_matrix:
                atom = Atom([term[i]
                             for i in range(0, pred.arity)], pred.predicate, strata=pred.strata)
                if atom not in added_atoms:
                    ground_atoms.append(atom)
                    added_atoms[atom] = 1
                if self.allow_negation and not(pred == self.language_frame.target):
                    neg_atom = negate_predicate(atom)
                    if neg_atom not in added_atoms:
                        ground_atoms.append(neg_atom)
                        added_atoms[neg_atom] = 1

        return ground_atoms

    def convert(self):
        '''Generate initial valuations
        '''
        ground_atoms = self.generate_ground_atoms()
        valuation_mapping = {}
        initial_valuation = []
        for idx, atom in enumerate(ground_atoms):
            if atom in self.background:
                initial_valuation.append(1)
                valuation_mapping[atom] = idx
            else:
                initial_valuation.append(0)
                valuation_mapping[atom] = idx
        return (tf.convert_to_tensor(initial_valuation, dtype=np.float32), valuation_mapping)
        #return (np.array(initial_valuation, dtype=np.float32), valuation_mapping)

    def create_negation_mapping(self,valuation_mapping):
        ''' Generate negation mapping
        '''
        negation_mapping_tmp1 = [] 
        negation_mapping_tmp2 = [] 
        vi = 0
        for vm in valuation_mapping:
          if vm._negated:
             vmi = 0
             for vmm in valuation_mapping:
               if vm._predicate[4:] == vmm._predicate  and vm.terms == vmm.terms:
                  negation_mapping_tmp1.append(vi)
                  negation_mapping_tmp2.append(vmi)
                  vmi += 1
                  break
               vmi += 1
          vi += 1

        return [negation_mapping_tmp1, negation_mapping_tmp2]

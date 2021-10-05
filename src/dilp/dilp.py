'''Defines the main differentiable ILP code
'''

from src.ilp import ILP, Program_Template, Language_Frame, Rule_Template, Inference
from src.ilp.generate_rules import Optimized_Combinatorial_Generator
from src.core import Clause
import tensorflow as tf
from collections import OrderedDict
import numpy as np
from src.utils import printProgressBar
#import tensorflow.contrib.eager as tfe # obsolete in TF2
import os
from time import time
import pickle

class DILP():

    def __init__(self, language_frame: Language_Frame, background: list, positive: list, negative: list, program_template: Program_Template, allow_negation=False, output_path='./', verbose=True, reg_weight=0.02):
        '''
        Arguments:
            language_frame {Language_Frame} -- language frame
            background {list} -- background assumptions
            positive {list} -- positive examples
            negative {list} -- negative examples
            program_template {Program_Template} -- program template
            allow_negation -- Create and use negated predicates
        '''
        self.language_frame = language_frame
        self.background = background
        self.positive = positive
        self.negative = negative
        self.program_template = program_template
        self.training_data = OrderedDict()  # index to label
        self.allow_negation = allow_negation
        self.output_path = output_path
        if not os.path.exists(output_path):
            print("Creating folder :", output_path)
            os.makedirs(output_path)
        self.verbose = verbose
        self.__init__parameters()
        self.set_stratas()
        self.set_vmask()
        self.regularization = 0
        self.reg_weight = 0

    def __init__parameters(self):
        self.rule_weights = OrderedDict()
        self.lr = 0.05
        ilp = ILP(self.language_frame, self.background,
                  self.positive, self.negative, self.program_template, self.allow_negation)
        (valuation, valuation_mapping) = ilp.convert()
        self.valuation_mapping = valuation_mapping
        if self.allow_negation:
            negation_mapping = ilp.create_negation_mapping(valuation_mapping)
            self.negation_mapping = negation_mapping
            #print(valuation_mapping)
            #print(negation_mapping)
        self.base_valuation = valuation
        self.current_valuation = valuation
        self.valuation_mask = tf.constant(np.ones(len(valuation)),dtype=tf.float32)
        self.deduction_map = {}
        self.clause_map = {}
        print("Initialization of dilp class ...")
        with tf.compat.v1.variable_scope("rule_weights", reuse=tf.compat.v1.AUTO_REUSE):
            for p in [self.language_frame.target] + self.program_template.p_a:
                print("Predicate :")
                print(p)
                print("\n")
                rule_manager = Optimized_Combinatorial_Generator(
                    self.program_template.p_a + [self.language_frame.target], self.program_template.rules[p], p, self.language_frame.p_e, self.allow_negation)
                generated = rule_manager.generate_clauses()
                # print(generated)                
                self.clause_map[p] = generated
                self.rule_weights[p] = tf.compat.v1.get_variable(p.predicate + "_rule_weights",
                                                       [len(generated[0]), len(
                                                           generated[1])],
                                                       initializer=tf.compat.v1.random_normal_initializer,
                                                       dtype=tf.float32)
                #print('Rule Weights .. for current predicate')
                #print(self.rule_weights[p])
                #print(self.rule_weights[p].shape)
                print("\n") 
                deduction_matrices = []
                elm1 = []
                for clause1 in generated[0]:
                    elm1.append(Inference.x_c(
                        clause1, valuation_mapping, self.language_frame.constants))
                elm2 = []
                for clause2 in generated[1]:
                    elm2.append(Inference.x_c(
                        clause2, valuation_mapping, self.language_frame.constants))
                deduction_matrices.append((elm1, elm2))
                self.deduction_map[p] = deduction_matrices
                     
        for atom in valuation_mapping:
            if atom in self.positive:
                self.training_data[valuation_mapping[atom]] = 1.0
            elif atom in self.negative:
                self.training_data[valuation_mapping[atom]] = 0.0

        # print clause map
        if self.verbose:
            self.print_clause_map()

    def set_stratas(self):
        self.stratas = OrderedDict()
        self.max_stratas = 0
        for predi, pred in enumerate(self.clause_map):
          self.stratas[pred]=[]
          for cmi, cm in enumerate(self.clause_map[pred]):
            #print(len(cm))
            self.stratas[pred].append([])
            for cli, cl in enumerate(cm):
              if not (cl == None) :
                self.stratas[pred][cmi].append([cl.head.strata])
                if self.max_stratas < cl.head.strata:
                    self.max_stratas = cl.head.strata
              else:
                self.stratas[pred][cmi].append([])

        self.step_strata = np.zeros((self.program_template.T),dtype=int)
        step_strata_skip = self.program_template.T/(1 + self.max_stratas)
        for st in range(self.max_stratas+1):
          print(st*step_strata_skip,step_strata_skip*(st+1))
          self.step_strata[int(st*step_strata_skip):int(step_strata_skip*(st+1))] = st


    def set_vmask(self):
        self.vmasks = []
        for cstrata in range(self.max_stratas+1):
            self.vmasks.append(self.get_vmask_for_strata(cstrata))


    def get_vmask_for_strata(self, strata):
        vm = np.zeros(len(self.base_valuation))
        vmc = np.ones(len(self.base_valuation))
        for k,vmm in enumerate(self.valuation_mapping):
          if vmm.strata <= strata:
              vm[k] = 1.0
              vmc[k] = 0.0
        return [tf.constant(vm,dtype=tf.float32), tf.constant(vmc,dtype=tf.float32)]

    def __all_variables(self):
        return [weights for weights in self.rule_weights.values()]


    def show_atoms(self, valuation):
        result = {}
        for atom in self.valuation_mapping:
            if atom in self.positive:
                print('%s Expected: 1 %.3f' %
                      (str(atom), valuation[self.valuation_mapping[atom]]))
            elif atom in self.negative:
                print('%s Expected: 0 %.3f' %
                      (str(atom), valuation[self.valuation_mapping[atom]]))

    def show_definition(self):
        for predicate in self.rule_weights:
            shape = self.rule_weights[predicate].shape
            rule_weights = tf.reshape(self.rule_weights[predicate], [-1])
            weights = tf.reshape(tf.nn.softmax(rule_weights)[:, None], shape)
            print('----------------------------')
            print(str(predicate))
            clauses = self.clause_map[predicate]
            pos = np.unravel_index(
                np.argmax(weights, axis=None), weights.shape)
            print(clauses[0][pos[0]])
            print(clauses[1][pos[1]])

            '''
            for i in range(len(indexes[0])):
                if(weights[indexes[0][i], indexes[1][i]] > max_weights):
                    max_weights = weights[indexes[0][i],
                                          indexes[1][i]] > max_weights
                print(clauses[0][indexes[0][i]])
                print(clauses[1][indexes[1][i]])
            '''
            print('----------------------------')

    def print_clause_map(self):
        for pred in self.clause_map:
          print('------------------------')
          print('Clause : Strata (head, body1, body2) : ')
          print('------------------------')
          print(pred, ':', 'Number of clauses for first, second predicate :', len(self.clause_map[pred][0]), ','  , len(self.clause_map[pred][1]))
          for cm in self.clause_map[pred]:
            print('------------------------')
            for cmi in cm:
              if  not(cmi == None):
                print(cmi, ' :: ', cmi.head.strata, ',', cmi.body[0].strata, ',', cmi.body[1].strata) #,
 
    @staticmethod
    def update_progress(progress):
        print('\r[{0}] {1}%'.format('#' * (int(progress) / 10), progress))

        

    def train(self, steps=501, name='test'):
        """
        :param steps:
        :param name:
        :return: the loss history
        """
        # str2weights = {str(key): value for key,
        #                value in self.rule_weights.items()}
        # if name:
        #     checkpoint = tf.train.Checkpoint(**str2weights)
        #     checkpoint_dir = "./model/" + name
        #     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        #     try:
        #         checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #     except Exception as e:
        #         print(e)
        losses = []
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr) #0.05)
        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr) #0.05)

        for i in range(steps):
            grads = self.grad()

            optimizer.apply_gradients(zip(grads, self.__all_variables()),
                                      global_step=tf.compat.v1.train.get_or_create_global_step())
            loss_avg = float(self.loss().numpy())
            losses.append(loss_avg)
            print("-" * 20)
            print("step " + str(i) + " loss is " + str(loss_avg) + " Regularization is " + str(self.regularization))
            if i % 10 == 0:
                # self.show_definition()
                self.show_atoms(self.deduction())
                self.show_definition()
                # if name:
                # checkpoint.save(checkpoint_prefix)
                # pd.Series(np.array(losses)).to_csv(name + ".csv")

        return losses


    def loss(self, batch_size=-1):
        labels = np.array(
            [val for val in self.training_data.values()], dtype=np.float32)
        keys = np.array(
            [val for val in self.training_data.keys()], dtype=np.int32)
        outputs = tf.gather(self.deduction(), keys)
        if batch_size > 0:
            index = np.random.randint(0, len(labels), batch_size)
            labels = labels[index]
            outputs = tf.gather(outputs, index)
        
        loss = -tf.reduce_mean(input_tensor=labels * tf.math.log(outputs + 10e-10) +
                                   (1 - labels) * tf.math.log(1 - outputs + 10e-10))
        return loss

    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss(-1)
            #weight_decay = 0.0
            # regularization = 0
            #for weights in self.__all_variables():
            #    weights = tf.nn.softmax(weights)
            #    regularization += tf.reduce_sum(input_tensor=tf.sqrt(weights)) * weight_decay
            # regularization += norm_entropy(self.rule_weights)
            self.regularization = DILP.norm_entropy(self.rule_weights)
            loss_value += self.reg_weight * self.regularization #regularization / len(self.__all_variables())
        return tape.gradient(loss_value, self.__all_variables())

#@tf.function
    def deduction(self):
        # takes background as input and return a valuation of target ground atoms
        valuation = self.base_valuation
        cstrata = 0 
        print('Performing Inference')
        steps_per_strata = self.program_template.T/(1 + self.max_stratas)
        for step in range(self.program_template.T): # Forward chaining
            #printProgressBar(step, self.program_template.T, prefix='Progress:',
            #                 suffix='Complete', length=50)
            valuation = self.inference_step(self.deduction_map, self.rule_weights, self.vmasks, valuation, self.step_strata[step]) 
            if self.allow_negation: # apply negation
                # valuation = self.vmasks[self.step_strata[step]][0] * DILP.apply_negation(valuation, self.negation_mapping) + self.vmasks[self.step_strata[step]][1] * valuation 
                valuation = DILP.apply_negation(valuation, self.negation_mapping)
                # print(valuation)
            self.current_valuation = valuation
        print('Inference Complete')
        return valuation

    @staticmethod 
    def apply_negation(valuation, negation_mapping):
        # valuation = tf.convert_to_tensor(tf.Variable(valuation)[self.negation_mapping[0]].assign(1 - valuation[self.negation_mapping[1]]))
        #for (dm1, dm2) in zip(self.negation_mapping[0], self.negation_mapping[1]):
        #   valuation = tf.convert_to_tensor(tf.Variable(valuation)[dm1].assign(0)) # (1 - valuation[dm2])*0.0))
        dm = []
        for dmm in negation_mapping[0]:
          dm.append([dmm])
        #neg_updates = tf.constant(1.0 - tf.gather(valuation,tf.constant(self.negation_mapping[1])))
        neg_updates = 1.0 - tf.gather(valuation,tf.constant(negation_mapping[1]))
        return tf.tensor_scatter_nd_update(valuation,tf.constant(dm), neg_updates)

    @staticmethod 
    def inference_step(deduction_map, rule_weights, vmasks, valuation, cstrata):
        deduced_valuation = tf.zeros(valuation.shape[0])
        # deduction_matrices = self.rules_manager.deducation_matrices[predicate]
        for predicate in deduction_map:
            if predicate.strata == cstrata:
              for matrix in deduction_map[predicate]:
                deduced_valuation += DILP.inference_single_predicate(
                    valuation, matrix, rule_weights[predicate], vmasks[cstrata])
        return deduced_valuation + valuation - deduced_valuation * valuation


    @tf.function(jit_compile=True)  #@staticmethod  
    def inference_single_predicate(valuation, deduction_matrices, rule_weights, vmasks):
        '''
        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        result_valuations = DILP.compute_next_valuation(deduction_matrices, valuation, vmasks)
        c_p = []  # flattened
        for clause1 in result_valuations[0]:
            for clause2 in result_valuations[1]:
                #c_p.append(prob_sum(clause1, clause2))
                c_p.append(tf.maximum(clause1, clause2))
        rule_weights = tf.reshape(rule_weights, [-1])
        prob_rule_weights = tf.nn.softmax(rule_weights)[:, None] 

        tmp_out = tf.reduce_sum(input_tensor=(tf.stack(c_p) * prob_rule_weights), axis=0) # This causes rounding of errors when used with prob_sum
        # tmp_out = tf.math.minimum(tmp_out,tf.ones_like(tmp_out))  --- This prevents 
        return tmp_out  
        

    @tf.function(jit_compile=True)  #@tf.function(experimental_compile=True)  
    def compute_next_valuation(deduction_matrices, valuation, vmasks):
        result_valuations = [[], []]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                # result_valuations[i].append(vmasks[0] * DILP.inference_single_clause(valuation, matrix) +
                #                             vmasks[1] * valuation)
                result_valuations[i].append(DILP.inference_single_clause(valuation, matrix))
        return result_valuations  

    @tf.function(jit_compile=True)  #@tf.function(experimental_compile=True) #
    def inference_single_clause(valuation, X):
        '''
        The F_c in the paper
        :param valuation:
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        #print(valuation.shape)
        X1 = X[:, :, 0, None]
        X2 = X[:, :, 1, None]
        Y1 = tf.gather_nd(params=valuation, indices=X1)
        Y2 = tf.gather_nd(params=valuation, indices=X2)
        Z = Y1 * Y2
        return tf.reduce_max(input_tensor=Z, axis=1)

    @staticmethod 
    def norm_entropy(rule_weights):
        norm_entropy = 0
        for p in rule_weights:
            pRW = tf.nn.softmax(tf.reshape(rule_weights[p],[-1]))
            norm_entropy += -tf.reduce_sum(pRW*tf.math.log(pRW)) / tf.math.log(tf.cast(pRW.shape[0], dtype=tf.float32))
        return norm_entropy / len(rule_weights)


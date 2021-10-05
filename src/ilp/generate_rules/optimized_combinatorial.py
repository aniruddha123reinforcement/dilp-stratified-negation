'''Optimized combinatorial class
'''

from src.ilp import Rule_Manger
from src.core import Atom, Term, Clause
from src.utils import check_neg_condition, negate_predicate

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Optimized_Combinatorial_Generator(Rule_Manger):

    def generate_clauses(self):
        '''Generate all clauses with some level of optimization
        '''
        rule_matrix = []
        for rule in self.rules:
            # logger.info('Generating clauses')
            if rule == None:
                rule_matrix.append([None])
                continue
            clauses = []
            intensional_predicates = []
            if (rule.recursive or rule.tail_recursion):
                p = list(set([self.target]))
            else:
                p = []
            if(rule.allow_intensional): 
                p = list(set(self.p_e + self.p_i + p))
                p_i = list(set(self.p_i))
                intensional_predicates = [atom.predicate for atom in p_i]
            else:
                p = list(set(self.p_e + p))
 
            variables = ['X_%d' %
                         i for i in range(0, self.target.arity + rule.v)]
            target_variables = ['X_%d' %
                                i for i in range(0, self.target.arity)]
            should_be_recursive = False
            should_have_extentional = False
            if rule.tail_recursion:
                should_be_recursive = True
                should_have_extentional = True
            print(p)
            # Generate the body list
            body_list = []
            head = Atom(
                [Term(True, var) for var in target_variables], self.target.predicate, self.target.strata)
            for var1 in variables:
                for var2 in variables:
                    if (not (rule.allow_same_variables)) and var1 == var2:
                        continue
                    else:
                        term1 = Term(True, var1)
                        term2 = Term(True, var2)
                        body_list.append([term1, term2])
            # Generate the list
            added_pred = {}
            for ind1 in range(0, len(p)):
                pred1 = p[ind1]
                for b1 in body_list:
                    for ind2 in range(ind1, len(p)):
                        pred2 = p[ind2]
                        for b2 in body_list:
                            new_clause = False
                            body1 = Atom([b1[index]
                                          for index in range(0, pred1.arity)], pred1.predicate, pred1.strata)
                            body2 = Atom([b2[index]
                                          for index in range(0, pred2.arity)], pred2.predicate, pred2.strata)

                            clause = Clause(head, [body1, body2])
                            # logger.info(clause)
                            # All variables in head should be in the body
                            #if not set(target_variables).issubset([v.name for v in b1] + [v.name for v in b2]):
                            if not set(target_variables).issubset(body1.variables.union(body2.variables)): 
                                continue
                            elif (rule.should_have_extentional or should_have_extentional) and (body1.variables.union(body2.variables)).issubset(target_variables):
                                continue
                            elif head == body1 or head == body2:  # No Circular
                                continue
                            # NOTE: Based on appendix requires to have a intensional predicate
                            elif rule.allow_intensional and not (body1.predicate in intensional_predicates or body2.predicate in intensional_predicates):
                                continue
                            elif (head.predicate == body1.predicate and head.variables == body1.variables) or (head.predicate == body2.predicate and head.variables == body2.variables):
                                continue
                            elif (rule.should_be_recursive or should_be_recursive) and not (body1.predicate == head.predicate or body2.predicate == head.predicate):
                                continue
                            elif rule.base_recursion and (body1.predicate == head.predicate or body2.predicate == head.predicate): # not (body1.predicate == body2.predicate):
                                continue
                            elif clause in added_pred:
                                continue
                            else:
                                #print(rule.variables_linked, set(body1.variables).issubset(body2.variables.union(head.variables)), not set(body2.variables).issubset(body1.variables.union(head.variables)))
                                new_clause = True
                                added_pred[clause] = 1
                                clauses.append(clause)
                                # print("Adding : ", clause)
                                if new_clause and self.allow_negation and rule.allow_negation:
                                    if check_neg_condition(body1, body2, head, self): #rule_manager):
                                        #print("Adding negation", body1, body2, nbody1, nbody2, body1.predicate, self.target.predicate)
                                        nbody1 = negate_predicate(body1)
                                        nbody2 = negate_predicate(body2)
                                        # Check safety
                                        if set(body2.variables).issubset(body1.variables) and not (body2.predicate == self.target.predicate):
                                          clause = Clause(head, [body1, nbody2])
                                          added_pred[clause] = 1
                                          clauses.append(clause)
                                        else:
                                          print("Unsafe ... : ", clause)
                                        if set(body1.variables).issubset(body2.variables) and not (body1.predicate == self.target.predicate):
                                          clause = Clause(head, [nbody1, body2])
                                          added_pred[clause] = 1
                                          clauses.append(clause)
                                        else:
                                          print("Unsafe ... : ", clause)
                                
            rule_matrix.append(clauses)
            # logger.info('Clauses Generated')
        return rule_matrix

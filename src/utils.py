'''Defines stateless utility functions
'''
from src.core import Atom
from copy import deepcopy

def is_intensional(atom: Atom):
    '''Checks if the atom is intensional. If true returns true, otherwise returns false

    Arguments:
        atom {Atom} -- Atom to be analyzed
    '''
    for term in atom.terms:
        if not term.isVariable:
            return False

    return True


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    total -= 1
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def check_neg_condition(body1, body2, head, rule_manager):
  ## Prevent 
  #if body1.predicate == rule_manager.target.predicate or body2.predicate == rule_manager.target.predicate:
  #   return False
  if (body1.strata > head.strata) or (body2.strata > head.strata):
      print("Rejecting due to stratification :", head, body1, body2,  str(head.strata), str(body1.strata), str(body2.strata))
      return False
  if body1.predicate == body2.predicate:
     ## Check if the variables are different
    for vi in range(body1.arity):
       if body1.terms[vi] != body2.terms[vi]:
          # print("stratification :", head, body1, body2, str(head.strata), str(body1.strata), str(body2.strata) )
           return True
       else:
           print("Rejecting due to unsafe :", head, body1, body2,  str(head.strata), str(body1.strata), str(body2.strata))
      
    return False 
  return True


def negate_predicate(pred):
  ptmp = deepcopy(pred)
  ptmp.predicate = 'neg_' + ptmp.predicate
  ptmp._negated = True 
  return ptmp

INTENSIONAL_REQUIRED_MESSAGE = 'Atom is not intensional'

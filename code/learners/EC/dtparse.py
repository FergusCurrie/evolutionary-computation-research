
from sklearn.tree import export_text
#from code.learners.EC.deap_extra import get_pset
from deap import gp

class SklearnParse:

    def __init__(self, pset):
        self.pset = pset

    def get_deap_primitive(self, s : str):
        assert(s in [x.name for x in self.pset.primitives[self.pset.ret]])
        index = [x.name for x in self.pset.primitives[self.pset.ret]].index(s)
        return self.pset.primitives[self.pset.ret][index]


    def get_deap_arg(self, i : int):
        assert(self.pset.terminals[self.pset.ret][0].name=='ARG0')
        return self.pset.terminals[self.pset.ret][i]

    def get_deap_terminal(self, n : int):
        index = [x.name for x in self.pset.terminals[self.pset.ret]].index(str(n))
        return self.pset.terminals[self.pset.ret][index]


    def patch_branch(self, b):
        """
        So branches have constistent size
        """
        patched = []
        depth = ""
        is_passed_depth = False
        for x in b.split():
            if '|' in x:
                depth += x
            else:
                if not is_passed_depth:
                    patched.append(depth)
                is_passed_depth = True
                patched.append(x)
        if 'feature' in b:
            assert(len(patched) == 4)
        if 'class' in b:
            assert(len(patched) == 3)
        j = ' '.join(patched)
        return j
            

    def patch_tree_depth(self, tree):
        """
        The tree depth splits wrong, patch them into one for each row and return 
        """
        return [self.patch_branch(branch) for branch in tree]

    def half_on_if(self, sklearn_tree):
        first_chunk = sklearn_tree[0]
        found_feature = False
        indent, feature, equality, mod = first_chunk.split()
        # Find index of second half of if 
        index = -1 
        for i,line in enumerate(sklearn_tree):
            if 'class' in line:
                continue
            indent2, feature2, equality2, mod2 = line.split()
            if (indent == indent2) and (feature == feature2) and (equality2 != equality) and (mod == mod2) and (index != 0):
                index = i
                break
        # Return
        return sklearn_tree[1:index], sklearn_tree[index+1:] # This should shave of the already used if 


    def calculate_center_from_class(self, class_val, n_classes):
        radius = 0.5 # size of the map prediction 
        lowest_threshold = 0.3 * n_classes * -1
        class_val = float(class_val)
        return lowest_threshold + (class_val * radius * 2) - radius

    def build_tree(self, sklearn_tree, n_unique, original_tree):
        """
        sklearn_tree is a list of string 
        
        Recursive tree building method from sklearn_tree
        """
        #breakpoint()
        assert(type(sklearn_tree[0] == str))
        assert(type(sklearn_tree == list))
        gptree = []
        # Base case to stop recursion
        if len(sklearn_tree) == 1:
            assert('class' in sklearn_tree[0])
            class_val = sklearn_tree[0][-1]
            assert(class_val.isdigit())
            center_class = self.calculate_center_from_class(class_val, n_unique)
            self.pset.addTerminal(center_class) # create new terminal 
            return [self.get_deap_terminal(center_class)] # some kind of switch statement here to select correct value for gp. 
        
        # Look at the first line of the tree
        first_chunk = sklearn_tree[0]
        first_chunk_split = first_chunk.split()
        
        feature = [x for x in first_chunk_split if 'feature' in x][0] # not possible if we 

        feature_primitive = self.get_deap_arg(int(feature.replace('feature_', '')))
        mod = float(first_chunk_split[-1])
        
        # Split case 
        
        if 'feature' in first_chunk: # this is an if statment a, b,c split
            # First deal with the 'a' in the three arguments to my_if 
            gptree.append(self.get_deap_primitive('my_if')) # first place if in 
            gptree.append(self.get_deap_primitive('sub'))   # subtract the mod from main
            gptree.append(feature_primitive)
            self.pset.addTerminal(mod) # create new terminal 
            gptree.append(self.get_deap_terminal(mod))
            
            # Split into 'b' and 'c' components
            first_half, second_half = self.half_on_if(sklearn_tree) # halves the arrays based upon first element if statement
            # Deal with the 'b' component (of gp) fist by using the second half of the sklearn if
            sec = self.build_tree(second_half, n_unique, original_tree)
            for x in sec:
                gptree.append(x)
                
            # Deal with the 'c' component (of gp) by using the first half of the sklearn if
            fir = self.build_tree(first_half, n_unique, original_tree)
            for x in fir:
                gptree.append(x)
            
        return gptree

    def sklearn_random_forest_to_deap_gp_pop(self, model, n_unique): # takes a rando
        pop = []
        for estimator in model.estimators_:
            text_representation = export_text(estimator)
            tree = [t for t in text_representation.split('\n')]
            p_tree =self.patch_tree_depth(tree) # make sure the trees have consistent height while being distinguishable. \
            p_tree.remove('')
            built_tree = self.build_tree(p_tree, n_unique, p_tree)
            pop.append(gp.PrimitiveTree(built_tree))
        return pop
            

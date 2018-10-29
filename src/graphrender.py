# @Patrick Siegler

import utils

#TODO bring order hack back?
ORDERHACK = False

class TraceGraphBuilderDOT:
    def __init__(self):
        self.str = 'digraph graphname\n{\nrankdir="LR"\n'
        self.current_layer = 0

    def add_layer(self, nodeLabels, name=None):
        n = len(nodeLabels)
        self.str += '{ rank=same; '
        if name is None:
            self.str += str(self.current_layer) + '[shape=plaintext, style="invis"]; '
        else:
            self.str += str(self.current_layer) + '[shape=plaintext, label="' + name + '"] '

        for i in range(n):
            self.str += '"' + str(self.current_layer) + '_' + str(i) + '" [label="' + str(nodeLabels[i]) + '"]'
        self.str += ' }\n'

        if (self.current_layer > 0):
            self.str += str(self.current_layer-1) + ' -> ' + str(self.current_layer) + '[style="invis"]\n'

        if ORDERHACK:
            if n > 0:
                self.str += '"' + str(self.current_layer) + '"'
                for i in range(n):
                    self.str += ' -> ' + '"' + str(self.current_layer) + '_' + str(i) + '"'
                self.str += '[style="invis"]\n'

        self.current_layer += 1

    def add_edge(self, src_layer, src_id, dest_layer, dest_id, label, weight=1):
        self.str += '"' + str(src_layer) + '_' + str(src_id) + '" -> "' + str(dest_layer) + '_' + str(dest_id) + '" [label="' + str(label) + '", penwidth=' + str(weight) + ', weight=' + str(int(round(weight,1)*10)) + ']\n'

    def write(self, name):
        self.str += '}'
        utils.makedirs(name)
        with open(name + '.dot', "w") as f:
            f.write(self.str)
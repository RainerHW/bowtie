# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import random
import io
import pdb

import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import prettyplotlib as ppl


# settings
indices = (0, 24, 49, 74, 99)


class Graph(nx.DiGraph):
    def __init__(self, graph=None):
        """Directed graph extended with bow tie functionality
        parameter graph must be a directed networkx graph"""
        super(Graph, self).__init__()
        self.lc_asp, self.lc_diam = 0, 0
        self.bow_tie, self.bow_tie_dict, self.bow_tie_changes = 0, 0, []
        self.bow_tie_nodes = 0
        if graph:
            self.add_nodes_from(graph)
            self.add_edges_from(graph.edges())

    def stats(self, prev_bow_tie_dict=None):
        """ calculate several statistical measures on the graphs"""
        # Core, In and Out
        cc = nx.strongly_connected_components(self)
        lc = self.subgraph(cc[0])
        scc = set(lc.nodes())
        scc_node = random.sample(scc, 1)[0]
        sp = nx.all_pairs_shortest_path_length(self)
        inc = {n for n in self.nodes() if scc_node in sp[n]}
        inc -= scc
        outc = set()
        for n in scc:
            outc |= set(sp[n].keys())
        outc -= scc

        # Tendrils, Tube and Other
        tube = set()
        out_tendril = set()
        in_tendril = set()
        other = set()

        remainder = set(self.nodes()) - scc - inc - outc
        inc_out = set()
        for n in inc:
            inc_out |= set(sp[n].keys())
        inc_out = inc_out - inc - scc - outc

        for n in remainder:
            if n in inc_out:
                if set(sp[n].keys()) & outc:
                    tube.add(n)
                else:
                    in_tendril.add(n)
            elif set(sp[n].keys()) & outc:
                out_tendril.add(n)
            else:
                other.add(n)
        self.bow_tie = [inc, scc, outc, in_tendril, out_tendril, tube, other]
        self.bow_tie_nodes = self.bow_tie
        #print ("Stats:\n inc is: %s\n scc is: %s\n out is: %s\n in_ten is: %s\n out_ten is: %s\n tube is: %s\n other is: %s\n" %(inc, scc, outc, in_tendril, out_tendril, tube, other))
        self.bow_tie = [100 * len(x)/len(self) for x in self.bow_tie]
        zipped = zip(['inc', 'scc', 'outc', 'in_tendril', 'out_tendril',
                      'tube', 'other'], range(7))
        c2a = {c: i for c, i in zipped}
        self.bow_tie_dict = {}
        for i, c in enumerate([inc, scc, outc, in_tendril, out_tendril, tube,
                               other]):
            for n in c:
                self.bow_tie_dict[n] = i

        if prev_bow_tie_dict:
            self.bow_tie_changes = np.zeros((len(c2a), len(c2a)))
            for n in self:
                self.bow_tie_changes[prev_bow_tie_dict[n],
                                     self.bow_tie_dict[n]] += 1
            self.bow_tie_changes /= len(self)


class GraphCollection(list):
    def __init__(self, label):
        super(GraphCollection, self).__init__()
        self.label = label

    def compute(self):
        """compute statistics on all the graphs in the collection
        the bow tie changes are only computed between selected indices,
        as indicated by the global variable indices"""
        bow_tie_dict = None
        for i, g in enumerate(self):
            if i in indices:
                g.stats(bow_tie_dict)
                bow_tie_dict = g.bow_tie_dict
            else:
                g.stats()


class Plotting(object):

    def __init__(self, graphs):
        self.graphs = graphs
        self.styles = ['solid', 'dashed']
        self.colors = ppl.colors.set2
        self.bounds = {}    
        #self.stackplot()
        #self.alluvial()

    def bowtieplot(self, bowtie_size):
        #print("bowtieplot called with size: %s") %bowtie_size
        # get graphs out of graph collection
        for i, gc in enumerate(self.graphs):
            data = [graph.bow_tie for graph in gc]
            for graph in gc:
                # remove axis legends
                plt.axis('off')
                fig = plt.gcf()
                ax = fig.gca()
                
                ax = self.setUpBackground(graph, ax)

                positions = {}
                positions.update(self.randomNodePositionsInTrapezes(graph, 'in'))
                positions.update(self.randomNodePositionsInTrapezes(graph, 'out'))

                nx.draw_networkx(graph, pos=positions, with_labels=False, ax=ax, nodelist=[5,6,7,8], edgelist=[])

                plt.savefig("plots/bowtie_vis_" + bowtie_size + ".png")
                plt.clf()
                #self.bounds = {}
                #print (graph.bow_tie_nodes)

        #print ("reached")

    def randomNodePositionsInTrapezes(self, graph, component):
        # todo: change depending on figure size and make constant

        if component == 'in':
            nodes = graph.bow_tie_nodes[0]
            in_left_x = self.bounds.get("in_left_x")
            in_left_y = self.bounds.get("in_left_y")
            in_right_x = self.bounds.get("in_right_x")
            in_right_y = self.bounds.get("in_right_y")    
        elif (component == 'out'):
            nodes = graph.bow_tie_nodes[2]
            in_left_x = self.bounds.get("out_left_x")
            in_left_y = self.bounds.get("out_left_y")
            in_right_x = self.bounds.get("out_right_x")
            in_right_y = self.bounds.get("out_right_y")

        max_points_horizontally = "200"
        max_points_vertically = "150"
        positions = {}

        # the 'height' of the Trapeze
        x_range = in_right_x - in_left_x
        # how many points can fit in it horizontally?
        points_in_range = np.floor(int(max_points_horizontally) * x_range)
        
        # get slope of the top leg of the trapeze
        slope_m = ((in_left_y - in_right_y) / (in_left_x - in_right_x))
        

        for n in nodes:
            # get random x coord for every point
            x_coord_unscaled = random.randint(0, points_in_range)
            # scale int down to floats
            x_coord = x_coord_unscaled / points_in_range * x_range + in_left_x
            
            # calculate the y bounds for given x in trapeze
            max_y_coord = slope_m * (x_coord - in_left_x) + in_left_y
            y_range = 2 * (max_y_coord - .5) # .5 is center_coordinate
            y_start = 1 - max_y_coord
            max_y_points = np.floor(int(max_points_vertically) * y_range)
            y_coord_unscaled = random.randint(0, max_y_points)
            y_coord = y_coord_unscaled / max_y_points * y_range + y_start
            positions[n] = np.array([x_coord, y_coord])

        return positions

    def setUpBackground(self, graph, ax):
        # component sizes in percent
        in_c = graph.bow_tie[0]
        scc_c = graph.bow_tie[1]
        out_c = graph.bow_tie[2]
        main_components = in_c + scc_c + out_c

        in_c = in_c / main_components
        scc_c = scc_c / main_components
        out_c = out_c / main_components

        bound_positions = {}

        # scc circle
        circle_radius = scc_c / 2
        center_coordinate = .5
        scc_circle = patches.Circle((center_coordinate, center_coordinate), circle_radius, facecolor='red')
        
        # IN-Trapeze
        bottom_left_x  = 0
        bottom_left_y  = .1 # todo: adjust according to size

        top_left_x  = bottom_left_x
        top_left_y  = bottom_left_y + (2 * (center_coordinate - bottom_left_y))
    
        bottom_right_x = center_coordinate - circle_radius + (circle_radius / 8) # addition for overlapping
        top_right_x = bottom_right_x

        top_right_y = 0.5 + np.sqrt(np.square(circle_radius) - np.square(bottom_right_x - 0.5))
        bottom_right_y = center_coordinate - (top_right_y - center_coordinate)

        in_trapeze_coordinates = np.array([[bottom_left_x, bottom_left_y],
                                        [bottom_right_x, bottom_right_y],
                                        [top_right_x, top_right_y],
                                        [top_left_x, top_left_y]])

        in_trapeze = patches.Polygon(in_trapeze_coordinates, closed=True, facecolor='yellow')
        
        # remember bounds for node placement
        bound_positions["in_left_x"] = top_left_x
        bound_positions["in_left_y"] = top_left_y
        bound_positions["in_right_x"] = top_right_x
        bound_positions["in_right_y"] = top_right_y

        # OUT-Trapeze
        bottom_right_x = 1
        bottom_right_y = .1 #todo: adjust according to size

        top_right_x = bottom_right_x
        top_right_y = bottom_right_y + (2 * (center_coordinate - bottom_right_y))

        bottom_left_x = center_coordinate + circle_radius - (circle_radius / 8)
        top_left_x = bottom_left_x

        top_left_y = 0.5 + np.sqrt(np.square(circle_radius) - np.square(bottom_left_x - 0.5)) #trouble?
        bottom_left_y = center_coordinate - (top_left_y - center_coordinate)

        out_trapeze_coordinates = np.array([[bottom_left_x, bottom_left_y],
                                        [bottom_right_x, bottom_right_y],
                                        [top_right_x, top_right_y],
                                        [top_left_x, top_left_y]])
        
        out_trapeze = patches.Polygon(out_trapeze_coordinates, closed=True, facecolor='blue')

        # remember bounds for node placement
        bound_positions["out_left_x"] = top_left_x
        bound_positions["out_left_y"] = top_left_y
        bound_positions["out_right_x"] = top_right_x
        bound_positions["out_right_y"] = top_right_y

        #add values to class variable
        self.bounds.update(bound_positions)

        ax.add_patch(in_trapeze)
        ax.add_patch(out_trapeze)
        ax.add_patch(scc_circle)


        return ax

    def stackplot(self):
        """produce stackplots for the graphcollections"""
        fig, axes = plt.subplots(1, len(self.graphs), squeeze=False,
                                 figsize=(8 * len(self.graphs), 6))

        legend_proxies = []
        for i, gc in enumerate(self.graphs):
            data = [graph.bow_tie for graph in gc]
            polys = axes[0, i].stackplot(np.arange(1, 1 + len(data)),
                                         np.transpose(np.array(data)),
                                         baseline='zero', edgecolor='face')
            legend_proxies = [plt.Rectangle((0, 0), 1, 1,
                              fc=p.get_facecolor()[0])
                              for p in polys]
        axes[0, -1].legend(legend_proxies, ['IN', 'SCC', 'OUT', 'TL_IN',
                           'TL_OUT', 'TUBE', 'OTHER'],
                           loc='upper center', bbox_to_anchor=(0.5, -0.1),
                           ncol=4, fancybox=True)

        # Beautification
        for col in range(axes.shape[0]):
            for row in range(axes.shape[1]):
                axes[col, row].set_ylabel('% of nodes')
                axes[col, row].set_ylim(0, 100)
                axes[col, row].set_xlim(1, 100)
                axes[col, row].set_xlabel('p in %')

        fig.subplots_adjust(left=0.08, bottom=0.21, right=0.95, top=0.95,
                            wspace=0.25, hspace=0.4)
        # plt.show()
        # save to disk
        fig.savefig('plots/bowtie_stacked.png')
        fig.savefig('plots/bowtie_stacked.pdf')

    def alluvial(self):
        """ produce an alluvial diagram (sort of like a flow chart) for the
        bowtie membership changes for selected indices,
        as indicated by the global variable indices"""
        ind = '    '  # indentation for the printed HTML and JavaScript files
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        dirpath = 'plots/alluvial/'
        with io.open(dirpath + 'alluvial.html', encoding='utf-8') as infile:
            template = infile.read().split('"data.js"')

        for i, gc in enumerate(self.graphs):
            data = [graph.bow_tie for graph in self.graphs[i]]
            changes = [g.bow_tie_changes
                       for j, g in enumerate(self.graphs[i]) if j in indices]
            fname = 'data_' + gc.label + '.js'
            with io.open(dirpath + fname, 'w', encoding='utf-8') as outfile:
                outfile.write('var data = {\n')
                outfile.write(ind + '"times": [\n')
                for iden, idx in enumerate(indices):
                    t = data[idx]
                    outfile.write(ind * 2 + '[\n')
                    for jdx, n in enumerate(t):
                        outfile.write(ind * 3 + '{\n')
                        outfile.write(ind * 4 + '"nodeName": "Node ' +
                                      unicode(jdx) + '",\n')
                        nid = unicode(iden * len(labels) + jdx)
                        outfile.write(ind * 4 + '"id": ' + nid +
                                      ',\n')
                        outfile.write(ind * 4 + '"nodeValue": ' +
                                      unicode(int(n * 100)) + ',\n')
                        outfile.write(ind * 4 + '"nodeLabel": "' +
                                      labels[jdx] + '"\n')
                        outfile.write(ind * 3 + '}')
                        if jdx != (len(t) - 1):
                            outfile.write(',')
                        outfile.write('\n')
                    outfile.write(ind * 2 + ']')
                    if idx != (len(data) - 1):
                        outfile.write(',')
                    outfile.write('\n')
                outfile.write(ind + '],\n')
                outfile.write(ind + '"links": [\n')

                for cidx, ci in enumerate(changes):
                    for mindex, val in np.ndenumerate(ci):
                        outfile.write(ind * 2 + '{\n')
                        s = unicode((cidx - 1) * len(labels) + mindex[0])
                        t = unicode(cidx * len(labels) + mindex[1])
                        outfile.write(ind * 3 + '"source": ' + s +
                                      ',\n')
                        outfile.write(ind * 3 + '"target": ' + t
                                      + ',\n')
                        outfile.write(ind * 3 + '"value": ' +
                                      unicode(val * 5000) + '\n')
                        outfile.write(ind * 2 + '}')
                        if mindex != (len(ci) - 1):
                            outfile.write(',')
                        outfile.write('\n')
                outfile.write(ind + ']\n')
                outfile.write('}')
            hfname = dirpath + 'alluvial_' + gc.label + '.html'
            with io.open(hfname, 'w', encoding='utf-8') as outfile:
                outfile.write(template[0] + '"' + fname + '"' + template[1])


if __name__ == '__main__':
    # create the graphs
    graphs = []
    gc = GraphCollection('gnp')
    for p in xrange(0, 100):
        g = nx.gnp_random_graph(10, p/100, directed=True)
        gc.append(Graph(g))
    graphs.append(gc)

    # compute statistics
    for g in graphs:
        g.compute()

    # plot
    P = Plotting(graphs)

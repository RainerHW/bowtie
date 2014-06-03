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
# canvas coordinates: bottom-left = (0,0); top-right = (1,1)
center_coordinate = 0.5


class Graph(nx.DiGraph):
    def __init__(self, graph=None):
        """
        Directed graph extended with bow tie functionality
        parameter graph must be a directed networkx graph
        """
        super(Graph, self).__init__()
        self.lc_asp, self.lc_diam = 0, 0
        self.bow_tie, self.bow_tie_dict, self.bow_tie_changes = 0, 0, []
        self.bow_tie_nodes = 0
        if graph:
            self.add_nodes_from(graph)
            self.add_edges_from(graph.edges())

    def stats(self, prev_bow_tie_dict=None):
        """
        calculate several statistical measures on the graphs
        """
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
        """
        compute statistics on all the graphs in the collection
        the bow tie changes are only computed between selected indices,
        as indicated by the global variable indices
        """
        bow_tie_dict = None
        for i, g in enumerate(self):
            if i in indices:
                g.stats(bow_tie_dict)
                bow_tie_dict = g.bow_tie_dict
            else:
                g.stats()


"""
Class for Plotting: bowtieplot, stackplot, alluvial
Argument:
- object: A list of GraphCollections, stored in self.graphs
"""
class Plotting(object):
    def __init__(self, graphs):
        self.graphs = graphs
        self.styles = ['solid', 'dashed']
        self.colors = ppl.colors.set2
        self.bounds = {}
        #self.stackplot()
        #self.alluvial()

    def bowtieplot(self, name, iteration):
        """
        Plots graphs as a bowtie
        Arguments:
        - name: of testcase, used for output-filename
        - iteration: extends the name in order to create animations
        """
        graph_counter = 1
        # get GraphCollection out of list
        for i, gc in enumerate(self.graphs):
            # get the graphs out of the graph collection: usually just one graph
            for graph in gc:
                # Remove Axis legends
                plt.axis('off')
                fig = plt.gcf()
                # get current axes
                ax = fig.gca()
                # make unscaleable and fixed bounds
                # necessary for steady animation
                ax.set_autoscale_on(False)
                ax.set_xbound(lower=0, upper=1)
                ax.set_ybound(lower=0, upper=1)
                
                # set up the Components (IN- & Out-Trapeze, SCC-Circle)
                ax = self.setUpComponentBackground(graph, ax)

                # node positions within the component backgrounds
                positions = {}
                positions.update(self.randomNodePositionsInTrapezes(graph, 'in'))
                positions.update(self.randomNodePositionInCircle(graph))
                positions.update(self.randomNodePositionsInTrapezes(graph, 'out'))

                # only plot the in, scc, and out nodes -> combine the sets
                relevantNodes = graph.bow_tie_nodes[0].union(graph.bow_tie_nodes[1].union(graph.bow_tie_nodes[2]))
                
                # create subgraph with only the relevant nodes
                subG = graph.subgraph(relevantNodes)
                
                # draw the graph/nodes; edges removed for clarity
                nx.draw_networkx(subG, pos=positions,
                                        with_labels=False,
                                        ax=ax,
                                        edgelist=[],
                                        node_size=100)
                # save to file using filename and iteration
                plt.savefig("plots/bowtie_vis_" + name + "_" + str(iteration).zfill(3) + "index_" + str(graph_counter).zfill(3) + ".png")
                plt.clf()

                # reset the bounds since multiple graphs can be in the collection
                self.bounds = {}
                graph_counter += 1

    def setUpComponentBackground(self, graph, ax):
        """
        Sets up the IN, OUT, and SCC component of the bow tie
        Argument:
        - graph: the graph to use
        - ax: the axes to plot on
        Returns:
        - ax: returns the axes which now holds the components
        """

        # component sizes in percent
        in_c = graph.bow_tie[0]
        scc_c = graph.bow_tie[1]
        out_c = graph.bow_tie[2]
        
        # we only use the main components
        main_components = in_c + scc_c + out_c

        # caluclate (new) percentage for each component
        in_c = in_c / main_components
        scc_c = scc_c / main_components
        out_c = out_c / main_components

        # save component bounds in dictionary for later use (node placement)
        bound_positions = {}

        # SCC-Circle
        # radius varies with size, should not overlap with label
        circle_radius = 0.05 + (scc_c / 3.5)
    
        # color varies with size, starting with light red increases intensity
        scaled_color_value = (250 - (250 * scc_c)) / 255 
        face_color_rgb = 1, scaled_color_value, scaled_color_value
        scc_circle = patches.Circle((center_coordinate, center_coordinate),
                                     circle_radius,
                                     facecolor=face_color_rgb)
        
        # IN-Trapeze coordinates
        # x starts at boarder, y varies with size
        in_bottom_left_x  = 0
        in_bottom_left_y  = 0.3 - (0.2 * in_c)

        # mirror over coordinate axis (which is as 0.5)
        in_top_left_x = in_bottom_left_x
        in_top_left_y = 1 - in_bottom_left_y
    
        # The bigger the SCC is, the smaller the overlap
        scc_overlap = circle_radius / 2  - (circle_radius / 2 * scc_c)
        
        # the right x-es
        in_bottom_right_x = center_coordinate - circle_radius + scc_overlap
        in_top_right_x = in_bottom_right_x

        # calculate intersection of trapeze and circle for the right x-es
        # using the Pythagorean theorem
        in_top_right_y = center_coordinate + np.sqrt(np.square(circle_radius) \
                         - np.square(in_bottom_right_x - center_coordinate))
        in_bottom_right_y = 1 - in_top_right_y

        # OUT-Trapeze coordiinates: like above, just mirrored
        out_bottom_right_x = 1
        out_bottom_right_y = 0.3 - (0.2 * out_c)

        out_top_right_x = out_bottom_right_x
        out_top_right_y = 1 - out_bottom_right_y

        out_bottom_left_x = center_coordinate + circle_radius - scc_overlap
        out_top_left_x = out_bottom_left_x

        out_top_left_y = center_coordinate + np.sqrt(np.square(circle_radius) - \
                         np.square(out_bottom_left_x - center_coordinate))
        out_bottom_left_y = 1 - out_top_left_y

        # create numpy arrays with coordinates to create polygon
        in_trapeze_coordinates = np.array([[in_bottom_left_x, in_bottom_left_y],
                                        [in_bottom_right_x, in_bottom_right_y],
                                        [in_top_right_x, in_top_right_y],
                                        [in_top_left_x, in_top_left_y]])

        out_trapeze_coordinates = np.array([[out_bottom_left_x, out_bottom_left_y],
                                        [out_bottom_right_x, out_bottom_right_y],
                                        [out_top_right_x, out_top_right_y],
                                        [out_top_left_x, out_top_left_y]])
        
        # settin up polygons, alpha depends on the size of the component
        # this makes bigger components more intense looking
        out_trapeze = patches.Polygon(out_trapeze_coordinates,
                                      closed=True,
                                      facecolor='green',
                                      alpha=out_c)

        in_trapeze = patches.Polygon(in_trapeze_coordinates,
                                     closed=True,
                                     facecolor='blue',
                                     alpha=in_c)
        
        # create Percentage Labels below components (if component > 0%)
        # font size depending on component sizes
        max_label_font_size = 60
        min_label_font_size = 10
        font_size_range = max_label_font_size - min_label_font_size
        
        # IN Component Label
        if (in_c):
            x_range = in_bottom_right_x - in_bottom_left_x
            in_label_x_coord = in_bottom_left_x + x_range / 2
            in_fontsize = min_label_font_size + (font_size_range * in_c)
            plt.text(in_label_x_coord, 0.02, str(int(in_c * 100)) + "%",
                     fontsize=in_fontsize, horizontalalignment='center')
        
        # OUT Component Label
        if (out_c):
            x_range = out_bottom_right_x - out_bottom_left_x
            out_label_x_coord = out_bottom_left_x + x_range / 2
            out_fontsize = min_label_font_size + (font_size_range * out_c)
            plt.text(out_label_x_coord, 0.02, str(int(out_c*100)) + "%",
                     fontsize=out_fontsize, horizontalalignment='center')

        # SCC Component Label
        if(scc_c):
            scc_fontsize = min_label_font_size + (font_size_range * scc_c)
            plt.text(center_coordinate, 0.02, str(int(scc_c*100)) + "%", 
                     fontsize=scc_fontsize, horizontalalignment='center')

        # remember bounds for node placement
        bound_positions["scc_circle_radius"] = circle_radius

        bound_positions["in_left_x"] = in_top_left_x
        bound_positions["in_left_y"] = in_top_left_y
        bound_positions["in_right_x"] = in_top_right_x
        bound_positions["in_right_y"] = in_top_right_y

        bound_positions["out_left_x"] = out_top_left_x
        bound_positions["out_left_y"] = out_top_left_y
        bound_positions["out_right_x"] = out_top_right_x
        bound_positions["out_right_y"] = out_top_right_y

        # add values to class variable
        self.bounds.update(bound_positions)

        ax.add_patch(in_trapeze)
        ax.add_patch(out_trapeze)
        ax.add_patch(scc_circle) 
        return ax

    def randomNodePositionsInTrapezes(self, graph, component):
        """
        Creates random positions within a trapeze
        Argument:
        - graph: the graph
        - component: either 'in' or 'out' trapeze
        Returns:
        - positions: a dictionary holding position-tuples for each node
        """
        # depending on component, the bounds of the specified trapeze are retrieved
        # note: it is a isosceles trapeze, therfore two coordinates suffice
        if (component == 'in'):
            nodes = graph.bow_tie_nodes[0]
            left_x = self.bounds.get("in_left_x")
            left_y = self.bounds.get("in_left_y")
            right_x = self.bounds.get("in_right_x")
            right_y = self.bounds.get("in_right_y")    
        elif (component == 'out'):
            nodes = graph.bow_tie_nodes[2]
            left_x = self.bounds.get("out_left_x")
            left_y = self.bounds.get("out_left_y")
            right_x = self.bounds.get("out_right_x")
            right_y = self.bounds.get("out_right_y")

        if len(nodes) is 0:
            return {}
        # define how many points fit onto the canvas
        max_points_horizontally = "200"
        max_points_vertically = "150"
        
        positions = {}
        # the 'height' of the Trapeze (the trapezes are lying on the side)
        x_range = right_x - left_x
        
        # how many points can fit in it horizontally?
        points_in_range = np.floor(int(max_points_horizontally) * x_range)
        
        # get slope of the top leg of the trapeze
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        # for every node in the component
        for n in nodes:
            # get random x coordinate
            x_coord_unscaled = random.randint(0, points_in_range)
            # scale int down to floats
            x_coord = x_coord_unscaled / points_in_range * x_range + left_x
            
            # calculate the y bounds for given x in trapeze
            # they vary for every x
            max_y_coord = slope_m * (x_coord - left_x) + left_y
            y_start = 1 - max_y_coord
            y_range = max_y_coord - y_start
            # scale max points to current range
            max_y_points = np.floor(int(max_points_vertically) * y_range)
            # get y unscaled coord
            y_coord_unscaled = random.randint(0, max_y_points)
            # scale it to our range
            y_coord = y_coord_unscaled / max_y_points * y_range + y_start
            # add position
            positions[n] = np.array([x_coord, y_coord])

        return positions

    def randomNodePositionInCircle(self, graph):
        """
        Creates random positions within the SCC-Circle
        Argument:
        - graph: the graph
        Returns:
        - positions: a dictionary holding position-tuples for each node
        """
        positions = {}
        
        # SCC Nodes
        nodes = graph.bow_tie_nodes[1]
        circle_radius = self.bounds.get('scc_circle_radius')
        
        # put nodes evenly distributed into circle
        for n in nodes:
            a = 2 * np.pi * random.random()
            r = np.sqrt(random.random())
            x_coord = (circle_radius * r) * np.cos(a) + center_coordinate #circle center position
            y_coord = (circle_radius * r) * np.sin(a) + center_coordinate #circle center position
            positions[n] = np.array([x_coord, y_coord])
        return positions

    def stackplot(self):
        """
        produce stackplots for the graphcollections
        [Created by Daniel Lamprecht]
        """
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
        """
        produce an alluvial diagram (sort of like a flow chart) for the
        bowtie membership changes for selected indices,
        as indicated by the global variable indices
        [Created by Daniel Lamprecht]
        """
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


"""
Main: Creates Erdös-Renyi graphs, computes the stats and plots them
[Created by Daniel Lamprecht]
"""
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

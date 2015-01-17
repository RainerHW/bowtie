# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals # todo comment
import random
import io
import pdb

import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as trsf
import prettyplotlib as ppl
from itertools import groupby
from operator import itemgetter
import graph_tool.all as gt
from PIL import Image as img

# settings
indices = (0, 0) #, 24, 49, 74, 99)
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
        # the following line does not work with nx 1.9.1 anymore
        lc = self.subgraph(cc[0])
        scc = set(lc.nodes())
        scc_node = random.sample(scc, 1)[0]
        sp = nx.all_pairs_shortest_path_length(self)
        # all nodes that reach the scc
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
                # out of bounds access when nodes are added to the graph
                # enable global indices again when working on it
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

    def bowtieplot(self, name):
        """
        Plots graphs as a bowtie
        Arguments:
        - name: of testcase, used for output-filename
        """
        # get GraphCollection out of list
        graphcounter = 0
        for i, gc in enumerate(self.graphs):
            # get the graphs out of the graph collection: usually just one graph
            for graph in gc:
                # create graph_tool graph
                g = gt.Graph()
                vList = g.add_vertex(len(graph.nodes()))
                g.add_edge_list(graph.edges())

                # only plot the in, scc, and out nodes -> combine the sets
                relevantNodes = graph.bow_tie_nodes[0].union(graph.bow_tie_nodes[1].union(graph.bow_tie_nodes[2]))
                # create subgraph with only the relevant nodes
                subG = graph.subgraph(relevantNodes)

                # and mask the graph_tool graph
                vprop_relevantNodes = g.new_vertex_property('boolean')
                for node in graph.nodes():
                    if (node in relevantNodes):
                        vprop_relevantNodes[g.vertex(node)]=bool(True)
                    else:
                        vprop_relevantNodes[g.vertex(node)]=bool(False)
                
                g.set_vertex_filter(vprop_relevantNodes)
                #print vprop_relevantEdges.key_type()
                #print vprop_relevantEdges.value_type()
                #print vprop_relevantEdges.get_graph()
                #print (vprop_relevantEdges)
                #print g
                # Remove Axis legends
                plt.axis('off') 
                fig = plt.gcf()
                
                # set file pixels: 1024x1024, default was 800x600: TODO: #define
                fig.set_size_inches(10.24, 10.24)
                
                # get current axis
                ax = fig.gca()
                # make unscaleable and fixed bounds
                ax.set_autoscale_on(False)
                ax.set_xbound(lower=0, upper=1)
                ax.set_ybound(lower=0, upper=1)

                # set padding for figure (TODO: needs to match graph_tool)
                figurePadding = np.array([[0.0, 0.0], [1, 1]])
                paddingBBox = trsf.Bbox(figurePadding)
                ax.set_position(paddingBBox)
                
                # calculate key nodes of graph
                keyNodes = self.findKeyNodes(graph)

                # set up the Components (IN- & Out-Trapeze, SCC-Circle)
                ax = self.setUpComponentBackground(graph, ax, keyNodes, percentLegend=False, nodeLegend=False, showSections=False)
                
                self.drawVerticalSectionLine(0.5)
                self.drawHorizontalSectionLine(0.25)
                self.drawHorizontalSectionLine(0.5)
                self.drawHorizontalSectionLine(0.75)
                
                # levels
                [in_levels, out_levels] = self.calcLevelsForNodes(g, graph.bow_tie_nodes, keyNodes)
                scc_levels = self.calcLevelsForSCC(g, graph.bow_tie_nodes, keyNodes)

                inNodePositions = self.calcPositionsForLevels(g, in_levels, 'in')
                outNodePositions = self.calcPositionsForLevels(g, out_levels, 'out')
                sccNodePositions = self.randomNodePositionInCircle(graph)
                
                #sccNodePositions = self.setUpSCCNodes(graph, keyNodes)
                # node positions within the component backgrounds
                positions = {}
                positions.update(sccNodePositions)
                positions.update(inNodePositions)
                positions.update(outNodePositions)
                
                # port positions to graph_tool graph
                vprop_positions = g.new_vertex_property("vector<float>")
                for idx in positions:
                    x_coord = positions[idx][0]
                    # need to swap the y, graph tool has the origin in the top left corner
                    y_coord = 1-positions[idx][1]
                    vprop_positions[g.vertex(idx)] = (x_coord, y_coord)

                # check canvas corners
                # adding graph_tool vertex at 1,1
                #v1 = g.add_vertex()
                #vprop_positions[g.vertex(v1)] = (1,1)

                # adding matplotlib rectangle at 1,1
                #rectangle = patches.Rectangle(xy=(0.99, 0.99), height=.01, width=.01, angle=0, color='r', fill='True', zorder=1)
                #rectangle2 = patches.Rectangle(xy=(0.99, 0), height=.01, width=.01, angle=0, color='r', fill='True', zorder=1)
                #ax.add_patch(rectangle)
                #ax.add_patch(rectangle2)

                graph_filename = "plots/bowtie_vis_" + name + "_" + str(graphcounter).zfill(3) + "_graph" + ".png"
                gt.graph_draw(g, pos=vprop_positions, vprops={"size": 32, "text": g.vertex_index, "font_size": 18}, fit_view=True, output_size=(1024, 1024), output=str(graph_filename))
                
                # draw the graph with nx
                nx.draw_networkx_nodes(subG, pos=positions, node_size=100)
                nx.draw_networkx_labels(subG, pos=positions, font_size=8)
                #nx.draw_networkx_edges(subG, pos=positions, arrows=True)

                # save to file using filename and graph number
                background_filename = "plots/bowtie_vis_" + name + "_" + str(graphcounter).zfill(3) + "_bg" + ".png"
                output_filename = "plots/bowtie_vis_" + name + "_" + str(graphcounter).zfill(3) + ".png"
                plt.savefig(background_filename)
                plt.clf()

                # overlay background and graph
                bgIMG = img.open(background_filename)
                gIMG = img.open(graph_filename)
                bgIMG.paste(gIMG, (0, 0), gIMG)
                bgIMG.save(output_filename)

                # reset the bounds since multiple graphs can be in the collection
                self.bounds = {}
                graphcounter += 1
                

                """
                labels = ['inc: ', 'scc: ', 'outc: ', 'in_tendril: ', 'out_tendril: ', 'tube: ', 'other: ']
                for label, component in zip(labels, graph.bow_tie_nodes):
                    print label + ", ".join(str(node) for node in component)
                """

    def calcLevelsForSCC(self, g, bow_tie_nodes, keyNodes):
        # [in_nodes_to_scc, scc_nodes_from_in, scc_nodes_to_out, scc_nodes_to_both, out_nodes_from_scc]
        scc_nodes_from_in = keyNodes[1]
        scc_nodes_to_out = keyNodes[2]
        scc_nodes_to_both = keyNodes[3]
        number_of_levels = 0
        #if (len())


    def calcLevelsForNodes(self, g, bow_tie_nodes, keyNodes):
        # todo: remove duplicated code --> no time improvement but readability (... maybe?!) ?
        in_nodes = bow_tie_nodes[0]
        out_nodes = bow_tie_nodes[2]
        scc_nodes_from_in = keyNodes[1]
        scc_nodes_to_out = keyNodes[2]

        # create property map for shortest path to ssc storage
        vprop_sp_in = g.new_vertex_property("int")
        vprop_sp_out = g.new_vertex_property("int")
        
        # calculate shortest path for every node in IN to every node in scc
        for scc_node in scc_nodes_from_in:
            for in_node in in_nodes:
                vlist, elist = gt.shortest_path(g, g.vertex(in_node), g.vertex(scc_node))
                if(len(elist) < vprop_sp_in[g.vertex(in_node)]  or vprop_sp_in[g.vertex(in_node)]  is 0):
                    vprop_sp_in[g.vertex(in_node)]  = len(elist)

        # calculate shortest path for every node in OUT to every node in scc
        for scc_node in scc_nodes_to_out:
            for out_node in out_nodes:
                vlist, elist = gt.shortest_path(g, g.vertex(scc_node), g.vertex(out_node))
                if(len(elist) < vprop_sp_out[g.vertex(out_node)]  or vprop_sp_out[g.vertex(out_node)]  is 0):
                    vprop_sp_out[g.vertex(out_node)]  = len(elist)

        # sections, equals the edges of the longest path
        in_sections = 0
        for in_node in in_nodes:
            if (vprop_sp_in[g.vertex(in_node)] > in_sections):
                in_sections = vprop_sp_in[g.vertex(in_node)]
        
        out_sections = 0
        for out_node in out_nodes:
            if (vprop_sp_out[g.vertex(out_node)] > out_sections):
                out_sections = vprop_sp_out[g.vertex(out_node)]

        # a list for every level; stored in the levels list
        in_levels = []
        for i in range (0, in_sections):
            in_levels.append(list())

        out_levels = []
        for i in range (0, out_sections):
            out_levels.append(list())

        # add nodes to levels
        for in_node in in_nodes:
            current_level = vprop_sp_in[g.vertex(in_node)]-1
            in_levels[current_level].append(in_node)

        for out_node in out_nodes:
            current_level = vprop_sp_out[g.vertex(out_node)]-1
            out_levels[current_level].append(out_node)

        """
        print "=============="
        print "In Node: " + str(in_node) + " to scc node: " + str(scc_node)
        print ([str(v) for v in vlist])
        print ([str(e) for e in elist])
        """
        # [in_nodes_to_scc, scc_nodes_from_in, scc_nodes_to_out, scc_nodes_to_both, out_nodes_from_scc]
        return [in_levels, out_levels]

    def calcPositionsForLevels(self, g, levels, component):
        positions = {}
        circle_radius = self.bounds.get('scc_circle_radius')
        if(component == 'in'):
            # trapeze corners
            left_x = self.bounds.get("in_left_x")
            left_y = self.bounds.get("in_left_y")
            right_x = self.bounds.get("in_right_x")
            right_y = self.bounds.get("in_right_y")
            
        elif (component == 'out'):
            left_x = self.bounds.get("out_left_x")
            left_y = self.bounds.get("out_left_y")
            right_x = self.bounds.get("out_right_x")
            right_y = self.bounds.get("out_right_y")
            # reverse the order, lowest level must be on top
            levels.reverse()
        
        # sections for trapeze
        sections = len(levels)
        # get slope of the top leg of the trapeze
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        if(component == 'in'):
            # use trapeze only until overlap with circle
            right_x = center_coordinate - circle_radius
            # need to recalculate right_y
            right_y = slope_m * (right_x - left_x) + left_y
        elif(component == 'out'):
            # check if "left_y = ..."" works
            left_x_new = center_coordinate + circle_radius
            left_y_new = slope_m * (left_x_new - left_x) + left_y
            left_x = left_x_new
            left_y = left_y_new
            self.drawVerticalSectionLine(left_x, 1-left_y)

        x_length = right_x - left_x
        for i in range(sections):
            # highest level (leftmost) to lowest
            levelNodes = levels.pop()
            # sections distributed evenly
            right_x = left_x + (x_length / sections)
            right_y = slope_m * (right_x - left_x) + left_y
            self.drawVerticalSectionLine(right_x, 1-right_y )
            # calculate positions in each section alone
            positions.update(self.calcTrapezeSectionPositions(levelNodes, left_x, right_x, left_y, right_y))
            left_x = right_x
            left_y = right_y
        return positions

    def calcTrapezeSectionPositions(self, nodes, left_x, right_x, left_y, right_y):
        circle_radius = self.bounds.get("scc_circle_radius")
        if len(nodes) is 0:
            return {}
        
        positions = {}
        # the 'height' of the Trapeze (the trapezes are lying on the side)
        x_range = right_x - left_x
        # use padding so nodes don't overlap
        x_padding = x_range / 8
        # reduce x_range for padding left and right
        x_range_padded = x_range - 2*x_padding

        # get slope of the top leg of the trapeze
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        # for every node in the component
        for index, node in enumerate(nodes):
            # x is in center of the sections
            x_coord = left_x + x_range/2
            # calculate the y bounds for given x in trapeze
            # they vary for every x
            max_y_coord = slope_m * (x_coord - left_x) + left_y
            y_start = 1 - max_y_coord
            y_range = max_y_coord - y_start
            
            # scale it to our range
            y_range_section = y_range / (len(nodes)+1)
            y_coord = y_start + (index+1) * y_range_section
            # add position
            positions[node] = np.array([x_coord, y_coord])

            # check wheter coord is also in scc circle (this can't happen anymore)
            # coord_in_scc_circle = np.power((x_coord - center_coordinate), 2) + \
            #                     np.power((y_coord - center_coordinate), 2) <= np.power(circle_radius, 2)
        return positions

    def drawVerticalSectionLine(self, x, y_padding=0):
        plt.plot((x, x), (y_padding, 1-y_padding), 'k--')

    def drawHorizontalSectionLine(self, y, x_padding=0):
        plt.plot((x_padding, 1-x_padding), (y,y), 'k--')
    
    def findKeyNodes(self, graph):
        in_nodes = graph.bow_tie_nodes[0]
        scc_nodes = graph.bow_tie_nodes[1]
        out_nodes = graph.bow_tie_nodes[2]
        in_ten_nodes = graph.bow_tie_nodes[3]
        out_ten_nodes = graph.bow_tie_nodes[4]
        tube_nods = graph.bow_tie_nodes[5]
        other_nodes = graph.bow_tie_nodes[6]
        in_nodes_to_scc = set()
        scc_nodes_from_in = set()
        scc_nodes_to_out = set()
        scc_nodes_to_both = set()
        out_nodes_from_scc = set()
        # get in-nodes that lead into scc
        for node in in_nodes:
            for edge in (graph.edges()):
                # edge starts in node? edge ends in scc?
                if node == edge[0] and edge[1] in scc_nodes:
                    # todo: count how many edges lead into scc
                    in_nodes_to_scc.add(node)
                    scc_nodes_from_in.add(edge[1])
        
        # get scc nodes that lead to out
        for node in scc_nodes:
            for edge in (graph.edges()):
                # edge starts in node? edge ends in out?
                if node == edge[0] and edge[1] in out_nodes:
                    # todo: count how many edges lead into out
                    if node not in scc_nodes_to_out:
                        scc_nodes_to_out.add(node)
                    if edge[1] not in out_nodes_from_scc:
                        out_nodes_from_scc.add(edge[1])
        
        # get scc nodes that have a connection to in and out 
        scc_nodes_to_both = scc_nodes_from_in.intersection(scc_nodes_to_out)
        #or are only connected to other nodes in scc
        all_to_in_and_out = scc_nodes_from_in.union(scc_nodes_to_out)
        
        """
        print "------------------------------------------------------"
        print 'scc from in: ' + str(scc_nodes_from_in)
        print 'scc to out: ' + str(scc_nodes_to_out)
        print 'scc both: ' + str(scc_nodes_to_both)
        print 'in to scc: ' + str(in_nodes_to_scc)
        print 'out from scc ' + str(out_nodes_from_scc)
        print "------------------------------------------------------"
        """
        # return array of arrays holding the key nodes of each component
        return [in_nodes_to_scc, scc_nodes_from_in, scc_nodes_to_out, scc_nodes_to_both, out_nodes_from_scc]

    def setUpComponentBackground(self, graph, ax, keyNodes, percentLegend=True, nodeLegend=False, showSections=True):

        """
        Sets up the IN, OUT, and SCC component of the bow tie
        Argument:
        - graph: the graph to use
        - ax: the axis to plot on
        Returns:
        - ax: returns the axis which now holds the components
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

        # store component defining points in dictionary (for later node calculations)
        bound_positions = {}

        # SCC-Circle: radius varies with size, should not overlap with label
        circle_radius = 0.05 + (scc_c / 3.5)
    
        # color varies with size, starting with light red increases intensity
        scaled_color_value = (250 - (250 * scc_c)) / 255 
        scc_face_color_rgb = 1, scaled_color_value, scaled_color_value
        scc_circle = patches.Circle((center_coordinate, center_coordinate),
                                     circle_radius,
                                     facecolor=scc_face_color_rgb,
                                     zorder=0.5) # zorder of edges is 1, move circle behind them

        # IN-Trapeze coordinates
        # x starts at y axis (x = 0), y varies with size
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
        
        # setting up polygons, alpha depends on the size of the component
        # this makes bigger components more intense looking
        out_trapeze = patches.Polygon(out_trapeze_coordinates,
                                      closed=True,
                                      facecolor='green',
                                      alpha=out_c,
                                      zorder=0)

        in_trapeze = patches.Polygon(in_trapeze_coordinates,
                                     closed=True,
                                     facecolor='blue',
                                     alpha=in_c,
                                     zorder=0)
        
        # create Percentage Labels below components (if component > 0%)
        # font size depending on component sizes
        max_label_font_size = 60
        min_label_font_size = 10
        font_size_range = max_label_font_size - min_label_font_size
        
        # Percentage labels at the bottom
        if (percentLegend):
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

	    # Legend: Which node in which component
        if(nodeLegend):
        	self.addNodeLegend(graph, plt, in_c, scc_face_color_rgb, out_c)

        if(showSections):
            #self.setUpSections(graph_tool_graph, keyNodes)
            # TODO: change to show section borders?
            a = 0

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

    def addNodeLegend(self, graph, plt, in_c, scc_face_color_rgb, out_c):
        # create legend (of nodes in components)
        inc_nodes_grouped_string = "inc: " + self.groupListItems(sorted(graph.bow_tie_nodes[0]))
        scc_nodes_grouped_string = "scc: " + self.groupListItems(sorted(graph.bow_tie_nodes[1]))
        out_nodes_grouped_string = "out: " + self.groupListItems(sorted(graph.bow_tie_nodes[2]))

        inc_patch = patches.Patch(color='blue', label=inc_nodes_grouped_string, alpha=in_c)
        scc_patch = patches.Patch(color=scc_face_color_rgb, label=scc_nodes_grouped_string)
        out_patch = patches.Patch(color='green', label=out_nodes_grouped_string, alpha=out_c)
        plt.legend(loc=2, frameon=False, borderpad=0, borderaxespad=0, handles=[inc_patch, scc_patch, out_patch])

    def groupListItems(self, sortedlistToGroup):
        """
        groups 3+ adjacent numbers in a list (e.g. [1, 2, 3, 4, 5, 8, 9]) to '1-4, 5, 8, 9'
        returns as string
        """
        groups = []
        for i, g in groupby(enumerate(sortedlistToGroup), lambda (i, x):i-x):
            groups.append(map(itemgetter(1), g))

        groupedItems = ""
        First = True
        for group in groups:
            # semicolons only between groups, not in front of first, not behind last
            if First:
                First = False
            else:
                groupedItems += ", "
            if len(group) > 2:
                groupedItems += str(group[0]) + "-" + str(group[-1])
            elif len(group) == 2:
                groupedItems += str(group[0]) + ", " + str(group[-1])
            else:
                groupedItems += str(group[0])
        return groupedItems


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
    
    """
    def randomNodePositionsInTrapezes(self, graph, component):
        # Creates random positions within a trapeze
        # Argument:
        # - graph: the graph
        # - component: either 'in' or 'out' trapeze
        # Returns:
        # - positions: a dictionary holding position-tuples for each node
        
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

        circle_radius = self.bounds.get("scc_circle_radius")
        positions = self.calcPositionsForTrapeze(nodes, left_x, right_x, left_y, right_y)
        return positions
    """

    """
    def setUpSCCNodes(self, graph, keyNodes):
        positions = {}
        scc_nodes_from_in = keyNodes[1]
        scc_nodes_to_out = keyNodes[2]
        scc_nodes_to_both = keyNodes[3]
        circle_radius = self.bounds.get("scc_circle_radius")
        dc = nx.degree_centrality(graph)
        bc = nx.betweenness_centrality(graph)
        #print(scc_nodes_from_in)
        #print(dc)
        #print(bc)
        
        # 0.0*PI < angle < 0.5*PI => top right circle quarter
        # 0.5*PI < angle < 1.0*PI => top left circle quarter
        # 1.0*PI < angle < 1.5*PI => bottom left circle quarter
        # 1.5*PI < angle < 2.0*PI => bottom right circle quarter
        

        all_nodes = scc_nodes_from_in.union(scc_nodes_to_both.union(scc_nodes_to_out))

        # todo: refactor !! think about better distributing (even distribution along the circumference?)
        for node in scc_nodes_from_in:
            # get random node between 0 and pi, then add pi/2 for left cirlce half
            angle = (random.randint(0, 99999999)*np.pi/99999999)+np.pi/2
            x = np.cos(angle)*circle_radius + center_coordinate
            y = np.sin(angle)*circle_radius + center_coordinate
            
            # move node towards center, depending on centralities
            node_distance_center = (center_coordinate - x, center_coordinate - y)
            node_position = (x + 2*dc[node] * node_distance_center[0], y + 1*bc[node] * node_distance_center[1])
            #rectangle = patches.Rectangle(xy=node_position, height=.01, width=.01, angle=0, color='r', fill='True', zorder=1)
            #plt.gcf().gca().add_patch(rectangle)
            positions[node] = np.array([node_position[0], node_position[1]])


        for node in scc_nodes_to_out:
            # get random node between 0 and pi, then add pi*1.5 for right cirlce half
            angle = (random.randint(0, 99999999)*np.pi/99999999)+np.pi*1.5
            x = np.cos(angle)*circle_radius + center_coordinate
            y = np.sin(angle)*circle_radius + center_coordinate
            
            # move node towards center, depending on centralities
            node_distance_center = (center_coordinate - x, center_coordinate - y)
            node_position = (x + 2*dc[node] * node_distance_center[0], y + 1*bc[node] * node_distance_center[1])
            positions[node] = np.array([node_position[0], node_position[1]])
        
        # Check balance of the graph
        left_side_node_count = len(scc_nodes_from_in) + len(graph.bow_tie_nodes[0])
        right_side_node_count = len(scc_nodes_to_out) + len(graph.bow_tie_nodes[2])
        left_side_ratio = left_side_node_count / (left_side_node_count + right_side_node_count)
        # value between -1 and 1, determining where the graph should lean
        graph_dynamics = 2 * (left_side_ratio - center_coordinate)

        for node in scc_nodes_to_both:
            y = random.randint(0, 99999999)*2*circle_radius/99999999 + center_coordinate - circle_radius
            x = center_coordinate
            # move node towards right depending on graph dynamics
            node_distance_to_rightmost = (circle_radius, center_coordinate - y)

            # move node towards center, depending on centralities
            node_distance_center = (center_coordinate - x, center_coordinate - y)
            random_factor = random.randint(0, 100)/100
            graph_dynamics = graph_dynamics * random_factor
            #node_position = (x, y + 1*bc[node] * node_distance_center[1])
            node_position = (x + graph_dynamics * node_distance_to_rightmost[0], y + graph_dynamics * node_distance_to_rightmost[1])
            positions[node] = np.array([node_position[0], node_position[1]])
        return positions
    """

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

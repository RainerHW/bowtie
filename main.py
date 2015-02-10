# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import random
import io
import pdb
from itertools import groupby
from operator import itemgetter
from PIL import Image as img

import numpy as np
import networkx as nx

# import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
plt.switch_backend('cairo')
import matplotlib.patches as patches
import matplotlib.transforms as trsf
from matplotlib.path import Path
#import prettyplotlib as ppl

import graph_tool.all as gt

# settings
indices = (0, 24, 49, 74, 99)
# canvas coordinates: bottom-left = (0,0); top-right = (1,1)
center_coordinate = 0.5


class Graph(gt.Graph):
    def __init__(self, graph=None):
        """
        Directed graph extended with bow tie functionality
        parameter graph must be a directed graph-tool graph
        """
        super(Graph, self).__init__()
        self.lc_asp, self.lc_diam = 0, 0
        self.bow_tie, self.bow_tie_dict, self.bow_tie_changes = 0, 0, []
        self.bow_tie_nodes = 0
        self.GT_PLACEHOLDER = 2147483647  # gt uses '2147483647' for unreachable nodes

    def stats(self, prev_bow_tie_dict=None):
        """
        calculate several statistical measures on the graphs
        """
        # list with nodes from largest_components
        largest_component = gt.label_largest_component(self, directed=True)
        # scc, In and Out
        scc = set()
        for vertex in self.vertices():
            if largest_component[vertex]:
                scc.add(int(vertex))
        scc_node = random.sample(scc, 1)[0]
        sp = gt.shortest_distance(self)
        # all nodes that reach the scc
        inc = set()
        for vertex in self.vertices():
            if sp[vertex][scc_node] != self.GT_PLACEHOLDER:
                inc.add(int(vertex))
        inc -= scc
        
        # all nodes from scc
        outc = set()
        for index in scc:
            for i, target_node in enumerate(sp[self.vertex(index)]):
                if target_node != self.GT_PLACEHOLDER:
                    outc.add(i)
        outc -= scc

        # Tendrils, Tube and Other
        tube = set()
        out_tendril = set()
        in_tendril = set()
        other = set()

        all_nodes = list(self.vertices())
        # convert all to ints
        all_nodes = [int(x) for x in all_nodes]
        remainder = set(all_nodes) - scc - inc - outc
        
        # out links in inc component
        inc_out = set()
        for index in inc:
            for i, target_node in enumerate(sp[self.vertex(index)]):
                if target_node != self.GT_PLACEHOLDER:
                    inc_out.add(i)
        inc_out = inc_out - inc - scc - outc

        for n in remainder:
            reachable_from_n = set()
            for i, target_node in enumerate(sp[self.vertex(n)]):
                if target_node != self.GT_PLACEHOLDER:
                    reachable_from_n.add(i)

            if n in inc_out:
                if reachable_from_n & outc:
                    tube.add(n)
                else:
                    in_tendril.add(n)
            elif reachable_from_n & outc:
                out_tendril.add(n)
            else:
                other.add(n)
        self.bow_tie = [inc, scc, outc, in_tendril, out_tendril, tube, other]
        self.bow_tie_nodes = self.bow_tie
        self.bow_tie = [100 * len(x)/len(list(self.vertices())) for x in self.bow_tie]
        zipped = zip(['inc', 'scc', 'outc', 'in_tendril', 'out_tendril',
                      'tube', 'other'], range(7))
        c2a = {c: i for c, i in zipped}
        self.bow_tie_dict = {}
        for i, c in enumerate([inc, scc, outc, in_tendril, out_tendril, tube, other]):
            for n in c:
                self.bow_tie_dict[n] = i
        # graph size of comparison must be the same (bow tie dics need to be of same size)
        if prev_bow_tie_dict and len(prev_bow_tie_dict) == len(self.bow_tie_dict):
            self.bow_tie_changes = np.zeros((len(c2a), len(c2a)))
            for n in self:
                self.bow_tie_changes[prev_bow_tie_dict[n], self.bow_tie_dict[n]] += 1
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
Class for Plotting: plot_bowtie, stackplot, alluvial
Argument:
- object: A list of GraphCollections, stored in self.graphs
"""
class Plotting(object):
    def __init__(self, graphs):
        self.graphs = graphs
        self.styles = ['solid', 'dashed']
        #self.colors = ppl.colors.set2
        self.trapeze_upper_corners = {}
        self.scc_circle_radius = 0
        self.component_settings = {}
        # key_nodes:
        #   in_nodes_to_scc
        #   scc_nodes_from_in
        #   scc_nodes_to_out
        #   scc_nodes_to_both
        #   out_nodes_from_scc
        self.key_nodes = {}
        self.sectionLines = []

        # matplotlib uses inches, graph_tool uses pixels (100dpi)
        self.screen_inches = 16.00

    def plot_bowtie(self, name):
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
                # only plot the in, scc, and out nodes -> combine the sets
                relevantNodes = graph.bow_tie_nodes[0].union(graph.bow_tie_nodes[1].union(graph.bow_tie_nodes[2]))
                # and mask the graph_tool graph
                vprop_relevantNodes = graph.new_vertex_property('boolean')
                for vertex in graph.vertices():
                    if int(vertex) in relevantNodes:
                        vprop_relevantNodes[vertex]=bool(True)
                    else:
                        vprop_relevantNodes[vertex]=bool(False)
                graph.set_vertex_filter(vprop_relevantNodes)
                
                # Set up Axis and Figure
                fig, ax = self.clear_figure(plt)
                # calculate key nodes of graph
                self.find_key_nodes(graph)

                # Add in- and out-trapeze (patch) and scc circle (patch) to axis
                ax, patches = self.draw_scc_in_out(graph, ax)
                self.show_component_node_legend(graph, plt)
                self.show_component_percent_legend()

                # we need to know which nodes are in which section/level of the component
                in_levels, out_levels = self.in_out_levels_holding_nodes(graph, graph.bow_tie_nodes)
                scc_levels = self.scc_levels_holding_nodes(graph, graph.bow_tie_nodes)
                # node positions within the component backgrounds
                
                # calculate positions for nodes
                vprop_positions = graph.new_vertex_property("vector<float>")
                #graph.vertex_properties[str("positions")] = graph.new_vertex_property("vector<float>")
                self.in_out_node_positions(graph, in_levels, 'in', vprop_positions)
                self.in_out_node_positions(graph, out_levels, 'out', vprop_positions)
                self.scc_node_positions(graph, scc_levels, vprop_positions)
                
                self.drawSectionLines()

                # create Tube # todo only show if tube component exists
                tubePatch = self.draw_tube(graph)
                ax.add_patch(tubePatch)

                # create Tendril(s)
                tendrilPatch1 = self.draw_tendril(graph, diameter=0.03, left_x=0.1)
                tendrilPatch2 = self.draw_tendril(graph, diameter=0.04, left_x=0.2)
                ax.add_patch(tendrilPatch1)
                ax.add_patch(tendrilPatch2)

                # concat filenames
                background_filename = "plots/bowtie_vis_" + name + "_" + str(graphcounter).zfill(3) + "_bg" + ".png"
                graph_filename = "plots/bowtie_vis_" + name + "_" + str(graphcounter).zfill(3) + "_graph" + ".png"
                output_filename = "plots/bowtie_vis_" + name + "_" + str(graphcounter).zfill(3) + ".png"
                #create files
                plt.savefig(background_filename, frameon=True, pad_inches=0)
                fig, ax = self.clear_figure(plt)
                # figure needs to be empty since zorder of gt-graph cant be changed
                gt.graph_draw(graph, pos=vprop_positions, \
                                                    vprops={"size": 32, "text": graph.vertex_index, "font_size": 18}, \
                                                    eprops={"marker_size":12}, \
                                                    output_size=(int(self.screen_inches*100), int(self.screen_inches*100)), \
                                                    fit_view=False, \
                                                    mplfig=fig)
                plt.savefig(graph_filename, frameon=True, transparent=True, pad_inches=0)
                # overlay background and graph
                background = img.open(background_filename)
                overlay = img.open(graph_filename)
                background.paste(overlay, (0, 0), overlay)
                background.save(output_filename)

                # reset stuff for next graph in collection
                plt.clf()
                self.trapeze_upper_corners = {}
                self.scc_circle_radius = 0
                self.sectionLines = []
                graphcounter += 1

    def clear_figure(self, plt):
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(self.screen_inches, self.screen_inches)
        ax.set_axis_off()
        #ax.set_autoscale_on(False)
        #ax.set_autoscalex_on(False)
        #ax.set_autoscaley_on(False)
        #ax.set_xbound(lower=0, upper=1)
        #ax.set_ybound(lower=0, upper=1)
        #ax.set_aspect(aspect='equal', adjustable='datalim')
        
        ax.autoscale(enable=False, axis='both', tight=True)
        ax.set_xlim(left=0, right=1, auto=False)
        ax.set_ylim(bottom=0, top=1, auto=False)
        
        # make figure use entire axis
        #figurePadding = np.array([[0.0, 0.0], [1, 1]])
        #paddingBBox = trsf.Bbox(figurePadding)
        #ax.set_position(paddingBBox)
        return fig, ax

    def scc_levels_holding_nodes(self, g, bow_tie_nodes):
        scc_nodes_from_in = self.key_nodes.get("scc_nodes_from_in")
        scc_nodes_to_out = self.key_nodes.get("scc_nodes_to_out")
        scc_nodes_to_both = self.key_nodes.get("scc_nodes_to_both")
        scc_nodes_from_in = scc_nodes_from_in.difference(scc_nodes_to_both) # todo: do this in find_keynodes
        scc_nodes_to_out = scc_nodes_to_out.difference(scc_nodes_to_both)

        # get number of sections
        number_of_levels = 0
        scc_levels = []
        if len(scc_nodes_from_in):
            number_of_levels += 1
            scc_levels.append(scc_nodes_from_in)
        if len(scc_nodes_to_both):
            number_of_levels += 1
            scc_levels.append(scc_nodes_to_both)
        if len(scc_nodes_to_out):
            number_of_levels += 1
            scc_levels.append(scc_nodes_to_out)

        # one section line needed
        if number_of_levels == 2:
            self.sectionLines.append([center_coordinate, center_coordinate - self.scc_circle_radius])

        if number_of_levels == 3:
            # divide circle into 3 sections of equal x-length
            center_offset = self.scc_circle_radius/3
            x_coord = center_coordinate - center_offset
            # find y coordinate on circumference for given x
            y = center_coordinate + np.sqrt((self.scc_circle_radius * self.scc_circle_radius) - np.power((x_coord - center_coordinate), 2))
            self.sectionLines.append([center_coordinate - center_offset, y])
            self.sectionLines.append([center_coordinate + center_offset, y])
        return scc_levels

    def scc_node_positions(self, graph, levels, vprop_positions):
        # calculate node positions
        if len(levels) == 1:
            self.scc_node_positions_random(graph, vprop_positions)
        elif len(levels) == 2:
            # align nodes vertically
            left_x = center_coordinate - (self.scc_circle_radius / 2)
            right_x = center_coordinate + (self.scc_circle_radius / 2)
            max_y_coord = center_coordinate + np.sqrt((self.scc_circle_radius * self.scc_circle_radius) - np.power((left_x - center_coordinate), 2))
            y_start = 1 - max_y_coord
            y_range = max_y_coord - y_start
            
            y_range_section = y_range / (len(levels[0])+1)
            for index, node in enumerate(levels[0]):
                y_coord = y_start + (index+1) * y_range_section
                vprop_positions[graph.vertex(node)] = np.array([left_x, y_coord])

            y_range_section = y_range / (len(levels[1])+1)
            for index, node in enumerate(levels[1]):
                y_coord = y_start + (index+1) * y_range_section
                vprop_positions[graph.vertex(node)] = np.array([right_x, y_coord])
        elif len(levels) == 3:
            x_offset = 2/3 * self.scc_circle_radius
            left_x = center_coordinate - x_offset
            middle_x = center_coordinate
            right_x = center_coordinate + x_offset

            middle_y_max = center_coordinate + self.scc_circle_radius
            middle_y_start = center_coordinate - self.scc_circle_radius
            y_range = 2 * self.scc_circle_radius

            y_range_section = y_range / (len(levels[1])+1)
            for index, node in enumerate(levels[1]):
                y_coord = middle_y_start + (index+1) * y_range_section
                vprop_positions[graph.vertex(node)] = np.array([middle_x, y_coord])

            left_right_y_max = center_coordinate + np.sqrt((self.scc_circle_radius * self.scc_circle_radius) - np.power((left_x - center_coordinate), 2))
            y_start = 1 - left_right_y_max
            y_range = left_right_y_max - y_start

            y_range_section = y_range / (len(levels[0])+1)
            for index, node in enumerate(levels[0]):
                y_coord = y_start + (index+1) * y_range_section
                vprop_positions[graph.vertex(node)] = np.array([left_x, y_coord])

            y_range_section = y_range / (len(levels[2])+1)
            for index, node in enumerate(levels[2]):
                y_coord = y_start + (index+1) * y_range_section
                vprop_positions[graph.vertex(node)] = np.array([right_x, y_coord])
        return vprop_positions

    def scc_node_positions_random(self, graph, vprop_positions):
        # SCC Nodes
        nodes = graph.bow_tie_nodes[1]
        # put nodes evenly distributed into circle
        for node in nodes:
            a = 2 * np.pi * random.random()
            r = np.sqrt(random.random())
            x_coord = (self.scc_circle_radius * r) * np.cos(a) + center_coordinate #circle center position
            y_coord = (self.scc_circle_radius * r) * np.sin(a) + center_coordinate #circle center position
            vprop_positions[graph.vertex(node)] = np.array([x_coord, y_coord])
        return
    
    def in_out_levels_holding_nodes(self, g, bow_tie_nodes):
        # todo: remove duplicated code --> no time improvement but readability (... maybe?!) ?
        in_nodes = bow_tie_nodes[0]
        out_nodes = bow_tie_nodes[2]
        
        scc_nodes_from_in = self.key_nodes.get("scc_nodes_from_in")
        scc_nodes_to_out = self.key_nodes.get("scc_nodes_to_out")

        # create property map for shortest path to ssc storage
        vprop_sp_in = g.new_vertex_property("int")
        vprop_sp_out = g.new_vertex_property("int")
        
        # calculate shortest path for every node in IN to every node in scc
        for scc_node in scc_nodes_from_in:
            for in_node in in_nodes:
                vlist, elist = gt.shortest_path(g, g.vertex(in_node), g.vertex(scc_node))
                if len(elist) < vprop_sp_in[g.vertex(in_node)]  or vprop_sp_in[g.vertex(in_node)]  is 0:
                    vprop_sp_in[g.vertex(in_node)]  = len(elist)

        # calculate shortest path for every node in OUT to every node in scc
        for scc_node in scc_nodes_to_out:
            for out_node in out_nodes:
                vlist, elist = gt.shortest_path(g, g.vertex(scc_node), g.vertex(out_node))
                if len(elist) < vprop_sp_out[g.vertex(out_node)]  or vprop_sp_out[g.vertex(out_node)]  is 0:
                    vprop_sp_out[g.vertex(out_node)]  = len(elist)

        # sections, equals the edges of the longest path
        in_sections = 0
        for in_node in in_nodes:
            if vprop_sp_in[g.vertex(in_node)] > in_sections:
                in_sections = vprop_sp_in[g.vertex(in_node)]
        
        out_sections = 0
        for out_node in out_nodes:
            if vprop_sp_out[g.vertex(out_node)] > out_sections:
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
        return in_levels, out_levels

    def in_out_node_positions(self, graph, levels, component, vprop_positions):
        if len(levels) == 0:
            return
        if component == 'in':
            # trapeze corners
            left_x = self.trapeze_upper_corners.get("in_left_x")
            left_y = self.trapeze_upper_corners.get("in_left_y")
            right_x = self.trapeze_upper_corners.get("in_right_x")
            right_y = self.trapeze_upper_corners.get("in_right_y")
        elif  component == 'out':
            left_x = self.trapeze_upper_corners.get("out_left_x")
            left_y = self.trapeze_upper_corners.get("out_left_y")
            right_x = self.trapeze_upper_corners.get("out_right_x")
            right_y = self.trapeze_upper_corners.get("out_right_y")
            # reverse the order, lowest level must be on top
            levels.reverse()
        else:
            return
        # sections for trapeze
        sections = len(levels)
        # get slope of the top leg of the trapeze
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        if component == 'in':
            # use trapeze only until overlap with circle
            right_x = center_coordinate - self.scc_circle_radius
            # need to recalculate right_y
            right_y = slope_m * (right_x - left_x) + left_y
        elif component == 'out':
            # check if "left_y = ..."" works
            left_x_new = center_coordinate + self.scc_circle_radius
            left_y_new = slope_m * (left_x_new - left_x) + left_y
            left_x = left_x_new
            left_y = left_y_new
            if sections > 1:
                self.sectionLines.append([left_x, 1-left_y])

        x_length = right_x - left_x
        for i in range(sections):
            # highest level (leftmost) to lowest
            levelNodes = levels.pop()
            # sections distributed evenly
            right_x = left_x + (x_length / sections)
            right_y = slope_m * (right_x - left_x) + left_y
            if sections > 1 and right_x != self.trapeze_upper_corners.get("out_right_x"):
                self.sectionLines.append([right_x, 1-right_y])
            # calculate positions in each section alone
            self.in_out_nodes_in_section_positions(graph, levelNodes, left_x, right_x, left_y, right_y, vprop_positions)
            left_x = right_x
            left_y = right_y
        return

    def in_out_nodes_in_section_positions(self, graph, nodes, left_x, right_x, left_y, right_y, vprop_positions):
        if len(nodes) is 0:
            return {}
        # the 'height' of the Trapeze (the trapezes are lying on the side)
        x_range = right_x - left_x

        # get slope of the top leg of the trapeze
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        # for every node in the component
        for index, node in enumerate(nodes):
            # x is in center of the sections
            x_coord = left_x + x_range/2
            # calculate the y trapeze_upper_corners for given x in trapeze
            # they vary for every x
            max_y_coord = slope_m * (x_coord - left_x) + left_y
            y_start = 1 - max_y_coord
            y_range = max_y_coord - y_start
            
            # scale it to our range
            y_range_section = y_range / (len(nodes)+1)
            y_coord = y_start + (index+1) * y_range_section
            # add position
            vprop_positions[graph.vertex(node)] = np.array([x_coord, y_coord])
        return

    def find_key_nodes(self, graph):
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
                if graph.vertex(node) == edge.source() and int(edge.target()) in scc_nodes:
                    in_nodes_to_scc.add(node)
                    scc_nodes_from_in.add(int(edge.target()))
        
        # get scc nodes that lead to out
        for node in scc_nodes:
            for edge in (graph.edges()):
                # edge starts in scc? edge ends in out?
                if graph.vertex(node) == edge.source() and int(edge.target()) in out_nodes:
                    scc_nodes_to_out.add(node)
                    out_nodes_from_scc.add(int(edge.target()))
        
        # get scc nodes that have a connection to in and out 
        scc_nodes_to_both = scc_nodes_from_in.intersection(scc_nodes_to_out)
        
        # or are only connected to other nodes in scc
        scc_nodes_to_both = scc_nodes_to_both.union(scc_nodes - scc_nodes_from_in - scc_nodes_to_out)
        
        # store in class dictionary
        self.key_nodes["in_nodes_to_scc"] = in_nodes_to_scc
        self.key_nodes["scc_nodes_from_in"] = scc_nodes_from_in
        self.key_nodes["scc_nodes_to_out"] = scc_nodes_to_out
        self.key_nodes["scc_nodes_to_both"] = scc_nodes_to_both
        self.key_nodes["out_nodes_from_scc"] = out_nodes_from_scc

    def draw_scc_in_out(self, graph, ax):
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

        # SCC-Circle: radius varies with size
        scc_circle_radius = 0.05 + (scc_c / 3.5)
    
        # color varies with size, starting with light red increases intensity
        scaled_color_value = (250 - (250 * scc_c)) / 255 
        scc_face_color = 1, scaled_color_value, scaled_color_value
        scc_circle = patches.Circle((center_coordinate, center_coordinate),
                                                     scc_circle_radius,
                                                     facecolor=scc_face_color,
                                                     edgecolor='#70707D',
                                                     linewidth=1.5,
                                                     zorder=0.5) # higher zorder gets drawn later
        # IN-Trapeze coordinates
        # x starts at y axis (x = 0), y varies with size
        in_bottom_left_x  = 0
        in_bottom_left_y  = 0.3 - (0.2 * in_c)

        # mirror over coordinate axis (which is as 0.5)
        in_top_left_x = in_bottom_left_x
        in_top_left_y = 1 - in_bottom_left_y
    
        # The bigger the SCC is, the smaller the overlap
        scc_overlap = scc_circle_radius / 2  - (scc_circle_radius / 2 * scc_c)
        
        # the right x-es
        in_bottom_right_x = center_coordinate - scc_circle_radius + scc_overlap
        in_top_right_x = in_bottom_right_x

        # calculate intersection of trapeze and circle for the right x-es
        # using the Pythagorean theorem
        in_top_right_y = center_coordinate + np.sqrt(np.square(scc_circle_radius) \
                         - np.square(in_bottom_right_x - center_coordinate))
        in_bottom_right_y = 1 - in_top_right_y

        # OUT-Trapeze coordiinates: like above, just mirrored
        out_bottom_right_x = 1
        out_bottom_right_y = 0.3 - (0.2 * out_c)

        out_top_right_x = out_bottom_right_x
        out_top_right_y = 1 - out_bottom_right_y

        out_bottom_left_x = center_coordinate + scc_circle_radius - scc_overlap
        out_top_left_x = out_bottom_left_x

        out_top_left_y = center_coordinate + np.sqrt(np.square(scc_circle_radius) - \
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
        
        scaled_color_value = (250 - (250 * scc_c)) / 255 
        in_face_color = scaled_color_value, 1, scaled_color_value
        in_trapeze = patches.Polygon(in_trapeze_coordinates,
                                                         closed=True,
                                                         facecolor=in_face_color,
                                                         edgecolor='#70707D',
                                                         linewidth=1.5,
                                                         zorder=0)
        scaled_color_value = (250 - (250 * scc_c)) / 255 
        out_face_color = scaled_color_value, scaled_color_value, 1
        out_trapeze = patches.Polygon(out_trapeze_coordinates,
                                                          closed=True,
                                                          facecolor=out_face_color,
                                                          edgecolor='#70707D',
                                                          linewidth=1.5,    
                                                          zorder=0)
        # store component defining points in dictionary (for later node calculations)
        self.scc_circle_radius = scc_circle_radius
        self.trapeze_upper_corners["in_left_x"] = in_top_left_x
        self.trapeze_upper_corners["in_left_y"] = in_top_left_y
        self.trapeze_upper_corners["in_right_x"] = in_top_right_x
        self.trapeze_upper_corners["in_right_y"] = in_top_right_y
        self.trapeze_upper_corners["out_left_x"] = out_top_left_x
        self.trapeze_upper_corners["out_left_y"] = out_top_left_y
        self.trapeze_upper_corners["out_right_x"] = out_top_right_x
        self.trapeze_upper_corners["out_right_y"] = out_top_right_y

        self.component_settings["scc_percentage"] = scc_c
        self.component_settings["in_percentage"] = in_c
        self.component_settings["out_percentage"] = out_c
        self.component_settings["scc_color"] = scc_face_color
        self.component_settings["in_color"] =  in_face_color
        self.component_settings["out_color"] =  out_face_color
        
        in_patch = ax.add_patch(in_trapeze)
        out_patch = ax.add_patch(out_trapeze)
        scc_patch = ax.add_patch(scc_circle)
        return ax, [in_patch, out_patch, scc_patch]

    def draw_tendril(self, graph, diameter=0.03, left_x=0.1):
        # get trapeze coordinates
        in_left_x = self.trapeze_upper_corners.get("in_left_x")
        in_left_y = self.trapeze_upper_corners.get("in_left_y")
        in_right_x = self.trapeze_upper_corners.get("in_right_x")
        in_right_y = self.trapeze_upper_corners.get("in_right_y")
        
        out_left_x = self.trapeze_upper_corners.get("out_left_x")
        out_left_y = self.trapeze_upper_corners.get("out_left_y")
        out_right_x = self.trapeze_upper_corners.get("out_right_x")
        out_right_y = self.trapeze_upper_corners.get("out_right_y")
        
        # trapeze top leg slopes
        slope_in = ((in_left_y - in_right_y) / (in_left_x - in_right_x))
        slope_out = ((out_left_y - out_right_y) / (out_left_x - out_right_x))

        tendril_left_x = left_x
        tendril_connect_left_y = in_left_y + slope_in * (tendril_left_x - in_left_x)

        tendril_right_x = tendril_left_x + diameter
        tendril_connect_right_y = in_left_y + slope_in * (tendril_right_x - in_left_x)

        # we don't want to go over the border (1.0) and we still need to close the tendril
        tendril_max_length = 0.9 - tendril_connect_left_y # todo: make 0.9 dynamic (= tendril length)

        tendril_mid_y = tendril_connect_left_y + tendril_max_length/2
        tendril_top_left_y = tendril_connect_left_y + tendril_max_length
        tendril_top_right_y = tendril_connect_left_y + tendril_max_length

        control_point_low_left_x = tendril_left_x - diameter
        control_point_low_right_x = tendril_right_x - diameter
        control_point_low_y = tendril_connect_left_y + (tendril_mid_y - tendril_connect_left_y)/2
        
        control_point_high_left_x = tendril_left_x + diameter
        control_point_high_right_x = tendril_right_x + diameter
        control_point_high_y = tendril_mid_y + (tendril_top_left_y - tendril_mid_y)/2

        control_point_top_left_x = tendril_left_x - diameter
        control_point_top_y = tendril_top_left_y + diameter
        control_point_top_right_x = tendril_right_x - diameter

        verts = [
            # left line tendril going up
            (tendril_left_x, tendril_connect_left_y),                       # left side of tendril connected to trapeze
            (control_point_low_left_x, control_point_low_y),        # curve tendril into one direction
            (tendril_left_x, tendril_mid_y),                                    # middle of tendril
            (control_point_high_left_x, control_point_high_y),     #curve into other direction
            (tendril_left_x, tendril_top_left_y),                              # reached top
            # build round top: todo: beautify
            (control_point_top_left_x, control_point_top_y),
            (control_point_top_right_x, control_point_top_y),
            (tendril_right_x, tendril_top_right_y), 
            # right line tendril going down
            (control_point_high_right_x, control_point_high_y),
            (tendril_right_x, tendril_mid_y),
            (control_point_low_right_x, control_point_low_y),
            (tendril_right_x, tendril_connect_right_y),
            # close poly
            (tendril_left_x, tendril_connect_left_y)
            ]

        codes = [
            # left line tendril going up
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            # round top
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            # right line tendril going down
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            # close poly
            Path.CLOSEPOLY
            ]
    
        path = Path(verts, codes)
        patch = patches.PathPatch(path, joinstyle='bevel', facecolor='#B2B2CC', edgecolor='#70707D', lw=1.5, zorder=0.4)
        return patch

    def draw_tube(self, graph):
        # get trapeze coordinates (but use the bottom side of trapeze -> '1-')
        in_left_x = self.trapeze_upper_corners.get("in_left_x")
        in_left_y = 1-self.trapeze_upper_corners.get("in_left_y")
        in_right_x = self.trapeze_upper_corners.get("in_right_x")
        in_right_y = 1-self.trapeze_upper_corners.get("in_right_y")
        
        out_left_x = self.trapeze_upper_corners.get("out_left_x")
        out_left_y = 1-self.trapeze_upper_corners.get("out_left_y")
        out_right_x = self.trapeze_upper_corners.get("out_right_x")
        out_right_y = 1-self.trapeze_upper_corners.get("out_right_y")
        
        slope_in = ((in_left_y - in_right_y) / (in_left_x - in_right_x))
        slope_out = ((out_left_y - out_right_y) / (out_left_x - out_right_x))

        # path starts/ends in middle of trapezes arms (trapezes have different arm lengths)
        line1_start_x = in_left_x + (in_right_x - in_left_x) / 2
        line1_start_y = in_left_y + slope_in * (line1_start_x - in_left_x)

        line1_end_x = out_left_x + (out_right_x - out_left_x) / 2
        line1_end_y = out_left_y + slope_out * (line1_end_x - out_left_x)

        # calculate bézier control point
        y_diff = max((in_right_y - line1_start_y), (out_left_y - line1_end_y))
        control_point_one = (center_coordinate, (center_coordinate - self.scc_circle_radius - y_diff))

        tube_diameter = 0.025 # make this dependent of size of tube component
        control_point_two = (center_coordinate, control_point_one[1]-tube_diameter)

        # calculate second line between trapezes, this time we start at right trapeze (path leads back)
        slope_out_y = ((out_left_x - out_right_x) / (out_left_y - out_right_y))
        line2_start_y = line1_end_y - tube_diameter
        line2_start_x = out_left_x + slope_out_y * (line2_start_y - out_left_y)

        slope_in_y = ((in_left_x - in_right_x) / (in_left_y - in_right_y))
        line2_end_y = line1_start_y - tube_diameter
        line2_end_x = in_left_x + slope_in_y * (line2_end_y - in_left_y)

        verts = [
            (line1_start_x, line1_start_y), # starts at in_trapeze
            control_point_one,                  # move below scc circle
            (line1_end_x, line1_end_y),   # ends at out trapeze
            (line2_start_x, line2_start_y), # starts at out trapeze (lower)
            control_point_two,                  # moves below control point 1
            (line2_end_x, line2_end_y),   # ends at in trapeze (lower than line1_start)
            (line1_start_x, line1_start_y)  # close path
            ]

        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CLOSEPOLY
            ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='#B2B2CC', edgecolor='#70707D', lw=1.5, zorder=0.4)
        return patch

    def draw_orientation_lines(self):
        self.draw_vertical_line(0.5)
        self.draw_horizontal_line(0.25)
        self.draw_horizontal_line(0.5)
        self.draw_horizontal_line(0.75)

    def draw_vertical_line(self, x, y_padding=0):
        plt.plot((x, x), (y_padding, 1-y_padding), 'k--')

    def draw_horizontal_line(self, y, x_padding=0):
        plt.plot((x_padding, 1-x_padding), (y,y), 'k--')

    def drawSectionLines(self):
        for line in self.sectionLines:
            self.draw_vertical_line(line[0], line[1])

    def show_component_node_legend(self, graph, plt):
        in_color = self.component_settings.get("in_color")
        scc_color = self.component_settings.get("scc_color")
        out_color = self.component_settings.get("out_color")

        # create legend (of nodes in components)
        inc_nodes_grouped_string = "inc: " + self.group_numbers(sorted(graph.bow_tie_nodes[0]))
        scc_nodes_grouped_string = "scc: " + self.group_numbers(sorted(graph.bow_tie_nodes[1]))
        out_nodes_grouped_string = "out: " + self.group_numbers(sorted(graph.bow_tie_nodes[2]))

        inc_patch = patches.Patch(label=inc_nodes_grouped_string, color=in_color)
        scc_patch = patches.Patch(label=scc_nodes_grouped_string, color=scc_color)
        out_patch = patches.Patch(label=out_nodes_grouped_string, color=out_color)
        plt.legend(loc=2, frameon=False, borderpad=0, borderaxespad=0, handles=[inc_patch, scc_patch, out_patch])

    def show_component_percent_legend(self):
        in_left_x = self.trapeze_upper_corners.get("in_left_x")
        in_right_x = self.trapeze_upper_corners.get("in_right_x")
        out_left_x = self.trapeze_upper_corners.get("out_left_x")
        out_right_x = self.trapeze_upper_corners.get("out_right_x")

        in_color = self.component_settings.get("in_color")
        scc_color = self.component_settings.get("scc_color")
        out_color = self.component_settings.get("out_color")

        in_c = self.component_settings.get("in_percentage")
        scc_c = self.component_settings.get("scc_percentage")
        out_c = self.component_settings.get("out_percentage")

        # create Percentage Labels below components if component > 0%
        # font size depending on component sizes
        max_label_font_size = 60
        min_label_font_size = 10
        font_size_range = max_label_font_size - min_label_font_size
        
        # Percentage labels at the bottom
        # IN Component Label
        if in_c:
            x_range = in_right_x - in_left_x
            in_label_x_coord = in_left_x + x_range / 2
            in_fontsize = min_label_font_size + (font_size_range * in_c)
            plt.text(in_label_x_coord, 0.02, str(int(in_c * 100)) + "%",
                     fontsize=in_fontsize, color=in_color)
        
        # SCC Component Label
        if scc_c:
            scc_fontsize = min_label_font_size + (font_size_range * scc_c)
            plt.text(center_coordinate, 0.02, str(int(scc_c*100)) + "%", 
                     fontsize=scc_fontsize, color=scc_color)

        # OUT Component Label
        if out_c:
            x_range = out_right_x - out_left_x
            out_label_x_coord = out_left_x + x_range / 2
            out_fontsize = min_label_font_size + (font_size_range * out_c)
            plt.text(out_label_x_coord, 0.02, str(int(out_c*100)) + "%",
                     fontsize=out_fontsize, color=out_color)
    def print_key_nodes_console(self):
        print 'in to scc: \t' + ", ".join(str(node) for node in self.key_nodes.get("in_nodes_to_scc"))
        print 'scc from in: \t' + ", ".join(str(node) for node in self.key_nodes.get("scc_nodes_from_in"))
        print 'scc to out: \t' + ", ".join(str(node) for node in self.key_nodes.get("scc_nodes_to_out"))
        print 'scc to both: \t' + ", ".join(str(node) for node in self.key_nodes.get("scc_nodes_to_both"))
        print 'out from scc \t' + ", ".join(str(node) for node in self.key_nodes.get("out_nodes_from_scc"))

    def print_component_nodes_console(self, graph):
        labels = ['inc: \t\t', 'scc: \t\t', 'outc: \t\t', 'in_tendril: \t', 'out_tendril: \t', 'tube: \t\t', 'other: \t\t']
        for label, component in zip(labels, graph.bow_tie_nodes):
            print label + ", ".join(str(node) for node in component)

    def group_numbers(self, sorted_list):
        """
        groups 3+ adjacent numbers in a list (e.g. [1, 2, 3, 4, 5, 8, 9]) to '1-4, 5, 8, 9'
        returns as string
        """
        groups = []
        for i, g in groupby(enumerate(sorted_list), lambda (i, x):i-x):
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

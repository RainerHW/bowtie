# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import random
import io
from itertools import groupby
from itertools import izip
from math import atan, sin, degrees
from operator import itemgetter
from os import remove as rmv
from PIL import Image as img

# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
plt.switch_backend('cairo')
import matplotlib.patches as patches
import matplotlib.transforms as trsf
import numpy as np
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
        # graph size must be the same for comparison (bowtie dics need to be of same size)
        if prev_bow_tie_dict and len(prev_bow_tie_dict) == len(self.bow_tie_dict):
            self.bow_tie_changes = np.zeros((len(c2a), len(c2a)))
            for node in list(self.vertices()):
                self.bow_tie_changes[prev_bow_tie_dict[n], self.bow_tie_dict[n]] += 1
            self.bow_tie_changes /= len(list(self.vertices()))

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
        for i, graph in enumerate(self):
            if i in indices:
                graph.stats(bow_tie_dict)
                bow_tie_dict = graph.bow_tie_dict
            else:
                graph.stats()

"""
Class for Plotting: plot_bowtie, stackplot, alluvial
Argument:
- object: A list of GraphCollections, stored in self.graphs
"""
class Plotting(object):
    def __init__(self, graphs):
        self.graphs = graphs
        self.styles = ['solid', 'dashed'] # todo: remove?
        #self.colors = ppl.colors.set2
        self.component_settings = {}
        self.key_nodes = {}
        self.scc_circle_radius = 0
        self.sectionLines = []
        self.trapezoid_upper_corners = {}
        self.tube_key_points = {}

        # matplotlib uses inches, graph_tool uses pixels (100dpi)
        self.screen_inches = 16.00

    def plot_bowtie(self, name, show_legends=True, show_sections=True, save_bg_file=True, save_graph_file=False, only_background=True):
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
                # Set up Axis and Figure
                fig, ax = self.clear_figure(plt, show_legends)
                # calculate key nodes of graph
                self.find_key_nodes(graph)

                # Add in- and out-trapezoid (patch) and scc circle (patch) to axis
                self.draw_scc_in_out(graph, ax)

                # we need to know which nodes are in which section/level of the component
                in_levels, out_levels = self.in_out_levels_holding_nodes(graph)
                scc_levels = self.scc_levels_holding_nodes(graph)
                
                # calculate positions for nodes
                vprop_positions = graph.new_vertex_property("vector<float>")
                self.in_out_node_positions(graph, in_levels, 'in', vprop_positions)
                self.in_out_node_positions(graph, out_levels, 'out', vprop_positions)
                self.scc_node_positions(graph, scc_levels, vprop_positions)
                
                if show_sections:
                    self.draw_section_lines()

                # create Tendril(s)
                if graph.bow_tie_nodes[3]:  # in_tendril
                    #ax.add_patch(self.draw_tendril(graph, diameter=0.03, position='left', component='in', max_height=0.9))
                    ax.add_patch(self.draw_tendril(graph, diameter=0.04, position='middle', component='in', max_height=0.8))
                    ax.add_patch(self.draw_tendril(graph, diameter=0.05, position='right', component='in', max_height=0.85))

                if graph.bow_tie_nodes[4]: # out_tendril
                    ax.add_patch(self.draw_tendril(graph, diameter=0.03, position='left', component='out', max_height=0.8))
                    ax.add_patch(self.draw_tendril(graph, diameter=0.04, position='middle', component='out', max_height=0.9))
                    ax.add_patch(self.draw_tendril(graph, diameter=0.05, position='right', component='out', max_height=0.85))

                # create Tube
                if graph.bow_tie_nodes[5]:  # tube-nodes
                    tubePatch = self.draw_tube(graph)
                    ax.add_patch(tubePatch)
                    draw_tube_nodes = True
                    if draw_tube_nodes:
                        self.tube_node_positions(graph, vprop_positions)
                        plot_tube = True

                # create Other (components)
                if graph.bow_tie_nodes[6]:
                    otherPatch = self.draw_other(graph)
                    # ax.add_patch(otherPatch)

                if show_legends:
                    self.show_component_node_legend(graph, plt) # add all nodes
                    self.show_component_percent_legend()
                    self.show_component_size_legend()

                # concat filenames
                filename_prefix = "plots/bowtie_" + name + "_" + str(graphcounter+1).zfill(2)
                background_filename =  filename_prefix + "_bg" + ".png"
                graph_filename = filename_prefix + "_graph" + ".png"
                output_filename = filename_prefix + ".png"
                #create files
                plt.savefig(background_filename, frameon=True, pad_inches=0) #todo: check if parameters needed
                fig, ax = self.clear_figure(plt, show_legends)
                
                # decide which nodes to plot
                self.nodes_to_plot(graph, inc=True, scc=True, outc=True, in_tendril=False, out_tendril=False, tube=plot_tube, other=False)
                
                # figure needs to be empty since zorder of gt-graph cant be changed
                gt.graph_draw(graph, pos=vprop_positions,
                                                    vprops={"size": 32, "text": graph.vertex_index, "font_size": 18},
                                                    eprops={"marker_size":12},
                                                    output_size=(int(self.screen_inches*100), int(self.screen_inches*100)),
                                                    fit_view=False,
                                                    mplfig=fig)
                plt.savefig(graph_filename, frameon=True, transparent=True, pad_inches=0)
                # overlay background and graph
                background = img.open(background_filename)
                overlay = img.open(graph_filename)
                background.paste(overlay, (0, 0), overlay)
                background.save(output_filename)

                # todo: don't generate files of not needed: draw graph on bg image immediately
                if save_graph_file is False:
                    rmv(graph_filename)
                if save_bg_file is False:
                    rmv(background_filename)

                # reset stuff for next graph in collection
                plt.clf()
                self.component_settings = {}
                self.key_nodes = {}
                self.scc_circle_radius = 0
                self.sectionLines = []
                self.trapezoid_upper_corners = {}

                graphcounter += 1

    def clear_figure(self, plt, show_legends):
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(self.screen_inches, self.screen_inches)
        ax.set_axis_on()
        ax.autoscale(enable=False, axis='both', tight=True)
        ax.set_xlim(left=0, right=1, auto=False)
        ax.set_ylim(bottom=0, top=1, auto=False)
        
        # make figure use entire axis
        if show_legends:
            # figurePadding = np.array([[0.0, 0.0], [1, 1]])
            figurePadding = np.array([[0.05, 0.05], [0.95, 0.95]])
            paddingBBox = trsf.Bbox(figurePadding)
            ax.set_position(paddingBBox)
        return fig, ax

    def find_key_nodes(self, graph):
        in_nodes = graph.bow_tie_nodes[0]
        scc_nodes = graph.bow_tie_nodes[1]
        out_nodes = graph.bow_tie_nodes[2]
        #in_ten_nodes = graph.bow_tie_nodes[3]
        #out_ten_nodes = graph.bow_tie_nodes[4]
        #tube_nodes = graph.bow_tie_nodes[5]
        #other_nodes = graph.bow_tie_nodes[6]
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
        
        # add those which are only connected to other nodes in scc
        scc_nodes_to_both = scc_nodes_to_both.union(scc_nodes - scc_nodes_from_in - scc_nodes_to_out)
        
        # remove scc_nodes_to_both from scc_nodes_from_in
        scc_nodes_from_in = scc_nodes_from_in.difference(scc_nodes_to_both)
        # and from out
        scc_nodes_to_out = scc_nodes_to_out.difference(scc_nodes_to_both)

        # store in class dictionary
        self.key_nodes["in_nodes_to_scc"] = in_nodes_to_scc
        self.key_nodes["scc_nodes_from_in"] = scc_nodes_from_in
        self.key_nodes["scc_nodes_to_out"] = scc_nodes_to_out
        self.key_nodes["scc_nodes_to_both"] = scc_nodes_to_both
        self.key_nodes["out_nodes_from_scc"] = out_nodes_from_scc

    def nodes_to_plot(self, graph, inc=True, scc=True, outc=True, in_tendril=False, out_tendril=False, tube=False, other=False):
        specified_nodes = set()
        if inc:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[0])
        if scc:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[1])
        if outc:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[2])
        if in_tendril:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[3])
        if out_tendril:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[4])
        if tube:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[5])
        if other:
            specified_nodes = specified_nodes.union(graph.bow_tie_nodes[6])
        
        vprop_specified_nodes = graph.new_vertex_property('boolean')
        for vertex in graph.vertices():
            if int(vertex) in specified_nodes:
                vprop_specified_nodes[vertex]=bool(True)
            else:
                vprop_specified_nodes[vertex]=bool(False)
        # apply mask to graph
        graph.set_vertex_filter(vprop_specified_nodes)

    def scc_levels_holding_nodes(self, graph):
        scc_nodes_from_in = self.key_nodes.get("scc_nodes_from_in")
        scc_nodes_to_out = self.key_nodes.get("scc_nodes_to_out")
        scc_nodes_to_both = self.key_nodes.get("scc_nodes_to_both")

        number_of_levels = 0
        # array holding array for each level with its nodes
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
        # calculate section lines
        self.scc_section_line_coords(number_of_levels)
        return scc_levels

    def scc_section_line_coords(self, number_of_levels):
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
    
    def in_out_levels_holding_nodes(self, graph):
        in_nodes = graph.bow_tie_nodes[0]
        out_nodes = graph.bow_tie_nodes[2]
        
        scc_nodes_from_in = self.key_nodes.get("scc_nodes_from_in")
        scc_nodes_to_out = self.key_nodes.get("scc_nodes_to_out")
        scc_nodes_to_both = self.key_nodes.get("scc_nodes_to_both")
        
        # 'center' scc nodes can have links to/from trapezoids as well --> add them
        scc_nodes_from_in = scc_nodes_from_in.union(scc_nodes_to_both)
        scc_nodes_to_out = scc_nodes_to_out.union(scc_nodes_to_both)

        # create property map for shortest path to ssc storage
        vprop_sp_in = graph.new_vertex_property("int")
        vprop_sp_out = graph.new_vertex_property("int")
        
        # calculate shortest path for every node in IN to every node in scc
        for scc_node in scc_nodes_from_in:
            for in_node in in_nodes:
                vlist, elist = gt.shortest_path(graph, graph.vertex(in_node), graph.vertex(scc_node))
                if len(elist) < vprop_sp_in[graph.vertex(in_node)] or vprop_sp_in[graph.vertex(in_node)] is 0:
                    vprop_sp_in[graph.vertex(in_node)] = len(elist)

        # calculate shortest path for every node in SCC to every node in OUT
        for scc_node in scc_nodes_to_out:
            for out_node in out_nodes:
                vlist, elist = gt.shortest_path(graph, graph.vertex(scc_node), graph.vertex(out_node))
                if len(elist) < vprop_sp_out[graph.vertex(out_node)] or vprop_sp_out[graph.vertex(out_node)] is 0:
                    vprop_sp_out[graph.vertex(out_node)] = len(elist)

        # number of sections, equals the edges of the longest path
        in_sections = vprop_sp_in.a.max()
        out_sections = vprop_sp_out.a.max()

        # a list for every level; stored in the levels list
        in_levels = []
        for i in range(0, in_sections):
            in_levels.append(list())

        out_levels = []
        for i in range(0, out_sections):
            out_levels.append(list())

        # add nodes to levels
        for in_node in in_nodes:
            current_level = vprop_sp_in[graph.vertex(in_node)]-1
            in_levels[current_level].append(in_node)

        for out_node in out_nodes:
            current_level = vprop_sp_out[graph.vertex(out_node)]-1
            out_levels[current_level].append(out_node)
        return in_levels, out_levels

    def in_out_node_positions(self, graph, levels, component, vprop_positions):
        if len(levels) == 0:
            return
        if component == 'in':
            # trapezoid corners
            left_x = self.trapezoid_upper_corners.get("in_left_x")
            left_y = self.trapezoid_upper_corners.get("in_left_y")
            right_x = self.trapezoid_upper_corners.get("in_right_x")
            right_y = self.trapezoid_upper_corners.get("in_right_y")
        elif  component == 'out':
            left_x = self.trapezoid_upper_corners.get("out_left_x")
            left_y = self.trapezoid_upper_corners.get("out_left_y")
            right_x = self.trapezoid_upper_corners.get("out_right_x")
            right_y = self.trapezoid_upper_corners.get("out_right_y")
            # reverse the order, lowest level must be on top
            levels.reverse()
        else:
            return
        # sections for trapezoid
        sections = len(levels)
        # get slope of the top leg of the trapezoid
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        if component == 'in':
            # use trapezoid only until overlap with circle
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
            if sections > 1 and (component == 'in' or (component == 'out' and i != (sections-1))): #right_x != self.trapezoid_upper_corners.get("out_right_x"):
                self.sectionLines.append([right_x, 1-right_y])
            # calculate positions in each section alone
            self.in_out_nodes_in_section_positions(graph, levelNodes, left_x, right_x, left_y, right_y, vprop_positions)
            left_x = right_x
            left_y = right_y
        return

    def in_out_nodes_in_section_positions(self, graph, nodes, left_x, right_x, left_y, right_y, vprop_positions):
        if len(nodes) is 0:
            return {}
        # the 'height' of the trapezoid (the trapezoids are lying on the side)
        x_range = right_x - left_x

        # get slope of the top leg of the trapezoid
        slope_m = ((left_y - right_y) / (left_x - right_x))
        
        # for every node in the component
        for index, node in enumerate(nodes):
            # x is in center of the sections
            x_coord = left_x + x_range/2
            # calculate the y trapezoid_upper_corners for given x in trapezoid
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

    def draw_scc_in_out(self, graph, ax):
        # component sizes in percent
        in_c = graph.bow_tie[0]
        scc_c = graph.bow_tie[1]
        out_c = graph.bow_tie[2]
        
        # we only use the main components
        main_components = in_c + scc_c + out_c

        # calculate (new) percentage for each component
        in_c = (in_c / main_components)
        scc_c = (scc_c / main_components)
        out_c = (out_c / main_components)

        # SCC-Circle: radius varies with size
        max_radius = 0.425
        min_radius = 0.05
        scc_circle_radius = min_radius + (scc_c * (max_radius - min_radius))

        # color varies with size, starting with light red increases intensity
        scaled_color_value = (250 - (250 * scc_c)) / 255 
        scc_face_color = 1, scaled_color_value, scaled_color_value
        scc_circle = patches.Circle((center_coordinate, center_coordinate),
                                                     scc_circle_radius,
                                                     facecolor=scc_face_color,
                                                     edgecolor='#70707D',
                                                     linewidth=1.5,
                                                     zorder=0.5) # higher zorder gets drawn later
        # store component defining points in dictionary (for later node calculations)
        self.scc_circle_radius = scc_circle_radius
        self.component_settings["scc_percentage"] = scc_c
        self.component_settings["scc_color"] = scc_face_color
        ax.add_patch(scc_circle)

        scc_area = np.power(scc_circle_radius, 2)*np.pi

        # The bigger the SCC is, the smaller the overlap with trapezoids
        scc_overlap = scc_circle_radius/2 - (scc_circle_radius / 2 * scc_c)

        if in_c:
            desired_in_area = scc_area * (in_c/scc_c)
            # IN-trapezoid coordinates
            in_right_x = center_coordinate - scc_circle_radius + scc_overlap
            # calculate corresponding y
            in_top_right_y = center_coordinate + np.sqrt(np.square(scc_circle_radius) -
                                                         np.square(in_right_x - center_coordinate))
            in_bottom_right_y = 1 - in_top_right_y
            in_left_x = 0
            in_top_left_y_max = 0.95
            in_top_left_y_min = in_top_right_y + 0.1
            # get maximum possible trapeze area
            in_max_area = self.trapezoid_without_circle_segment_area(in_left_x, in_top_left_y_max, in_right_x, in_top_right_y, 'in')
            in_min_area = self.trapezoid_without_circle_segment_area(in_left_x, in_top_left_y_min, in_right_x, in_top_right_y, 'in')
            # scc overlapped segment
            segment_height = in_right_x - (center_coordinate - self.scc_circle_radius)
            scc_circle_segment = self.circle_segment_area(segment_height, (in_top_right_y-in_bottom_right_y))
            a = in_top_right_y - in_bottom_right_y

            #print "areas: ", in_min_area, desired_in_area, in_max_area
            # find correct trapezoid size
            if in_min_area < desired_in_area < in_max_area:
                # find desired length of c of trapezoid
                trapezoids_height = in_right_x - in_left_x
                c = (2 * (desired_in_area + scc_circle_segment)/trapezoids_height) - a
                in_top_left_y = center_coordinate + (c/2)
                in_bottom_left_y = center_coordinate - (c/2)
            # otherwise we need to adjust the x-coord of the trapezoid
            elif desired_in_area < in_min_area:
                # find new h for trapezoid by using divide an conquer
                epsilon = 0.0000001
                emergency_exit = 1
                in_top_left_y = in_top_left_y_min
                x_jump = in_right_x/2
                slope = ((in_top_left_y - in_top_right_y) / (in_left_x - in_right_x))
                while True:
                    in_area = self.trapezoid_without_circle_segment_area(in_left_x, in_top_left_y, in_right_x, in_top_right_y, 'in')
                    emergency_exit += 1
                    if (desired_in_area-epsilon) < in_area < (desired_in_area+epsilon):
                        break
                    if in_area > desired_in_area:
                        in_left_x += x_jump
                        in_top_left_y += slope * x_jump
                        x_jump /= 2
                    elif in_area < desired_in_area:
                        in_left_x -= x_jump
                        in_top_left_y -= slope * x_jump
                        x_jump /= 2
                in_bottom_left_y = 1 - in_top_left_y
            else:
                print "ERROR: SCC IS TOO BIG!"
                print "desired_in: ", desired_in_area
                print "max_in: ", in_max_area
                print "scc_area: ", scc_area
                in_top_left_y = in_top_left_y_max
                in_bottom_left_y = 1-in_top_left_y
            #print "IN: desired vs. actual: ", desired_in_area, (self.trapezoid_without_circle_segment_area(in_left_x, in_top_left_y, in_right_x, in_top_right_y, 'in'))

            # create numpy arrays with coordinates to create polygon
            in_trapezoid_coordinates = np.array([[in_left_x, in_bottom_left_y],
                                                 [in_right_x, in_bottom_right_y],
                                                 [in_right_x, in_top_right_y],
                                                 [in_left_x, in_top_left_y]])

            scaled_color_value = (250 - (250 * in_c)) / 255
            in_face_color = scaled_color_value, 1, scaled_color_value
            in_trapezoid = patches.Polygon(in_trapezoid_coordinates,
                                           closed=True,
                                           facecolor=in_face_color,
                                           edgecolor='#70707D',
                                           linewidth=1.5,
                                           zorder=0)
            self.trapezoid_upper_corners["in_left_x"] = in_left_x
            self.trapezoid_upper_corners["in_left_y"] = in_top_left_y
            self.trapezoid_upper_corners["in_right_x"] = in_right_x
            self.trapezoid_upper_corners["in_right_y"] = in_top_right_y
            self.component_settings["in_percentage"] = in_c
            self.component_settings["in_color"] =  in_face_color
            ax.add_patch(in_trapezoid)
        if out_c:
            desired_out_area = scc_area * (out_c/scc_c)

            out_left_x = center_coordinate + scc_circle_radius - scc_overlap
            # calculate corresponding y
            out_top_left_y = center_coordinate + np.sqrt(np.square(scc_circle_radius) -
                                                         np.square(out_left_x - center_coordinate))
            out_bottom_left_y = 1 - out_top_left_y
            out_right_x = 1
            out_top_right_y_max = 0.95
            out_top_right_y_min = out_top_left_y + 0.1
            # get maximum possible trapeze area
            out_min_area = self.trapezoid_without_circle_segment_area(out_left_x, out_top_left_y, out_right_x, out_top_right_y_min, 'out')
            out_max_area = self.trapezoid_without_circle_segment_area(out_left_x, out_top_left_y, out_right_x, out_top_right_y_max, 'out')
            # scc overlapped segment
            segment_height = (center_coordinate + self.scc_circle_radius) - out_left_x
            scc_circle_segment = self.circle_segment_area(segment_height, (out_top_left_y-out_bottom_left_y))
            a = out_top_left_y - out_bottom_left_y

            # find correct trapezoid size
            if out_min_area < desired_out_area < out_max_area:
                # find desired length of c of trapezoid
                trapezoids_height = out_right_x - out_left_x
                c = (2 * (desired_out_area + scc_circle_segment)/trapezoids_height) - a
                out_top_right_y = center_coordinate + (c/2)
                out_bottom_right_y = center_coordinate - (c/2)
            # otherwise we need to adjust the x-coord of the trapezoid
            elif desired_out_area < out_min_area:
                # find new h for trapezoid by using divide an conquer
                epsilon = 0.0000001
                emergency_exit = 1
                out_top_right_y = out_top_right_y_min
                x_jump = (out_right_x - out_left_x)/2
                slope = ((out_top_right_y - out_top_left_y) / (out_right_x - out_left_x))
                while True:
                    out_area = self.trapezoid_without_circle_segment_area(out_left_x, out_top_left_y, out_right_x, out_top_right_y, 'out')
                    emergency_exit += 1
                    if (desired_out_area-epsilon) < out_area < (desired_out_area+epsilon):
                        break
                    if out_area > desired_out_area:
                        out_right_x -= x_jump
                        out_top_right_y -= slope * x_jump
                        x_jump /= 2
                    elif out_area < desired_out_area:
                        out_right_x += x_jump
                        out_top_right_y += slope * x_jump
                        x_jump /= 2
                out_bottom_right_y = 1 - out_top_right_y
            else:
                print "ERROR: SCC IS TOO BIG!"
                print "desired_out: ", desired_out_area
                print "max_out: ", out_max_area
                print "scc_area: ", scc_area
            #print "OUT: desired vs. actual: ", desired_out_area, (self.trapezoid_without_circle_segment_area(out_left_x, out_top_left_y, out_right_x, out_top_right_y, 'out'))

            out_trapezoid_coordinates = np.array([[out_left_x, out_bottom_left_y],
                                                  [out_right_x, out_bottom_right_y],
                                                  [out_right_x, out_top_right_y],
                                                  [out_left_x, out_top_left_y]])
            scaled_color_value = (250 - (250 * out_c)) / 255
            out_face_color = scaled_color_value, scaled_color_value, 1
            out_trapezoid = patches.Polygon(out_trapezoid_coordinates,
                                            closed=True,
                                            facecolor=out_face_color,
                                            edgecolor='#70707D',
                                            linewidth=1.5,
                                            zorder=0)
            self.trapezoid_upper_corners["out_left_x"] = out_left_x
            self.trapezoid_upper_corners["out_left_y"] = out_top_left_y
            self.trapezoid_upper_corners["out_right_x"] = out_right_x
            self.trapezoid_upper_corners["out_right_y"] = out_top_right_y
            self.component_settings["out_percentage"] = out_c
            self.component_settings["out_color"] = out_face_color
            ax.add_patch(out_trapezoid)
        return

    def draw_tendril(self, graph, component, diameter=0.03, position='middle', max_height=0.9):
        # get upper leg of corresponding component
        if component == 'in':
            left_x = self.trapezoid_upper_corners.get("in_left_x")
            left_y = self.trapezoid_upper_corners.get("in_left_y")
            right_x = self.trapezoid_upper_corners.get("in_right_x")
            right_y = self.trapezoid_upper_corners.get("in_right_y")
            ten_color = "#B2B2CC"
            self.component_settings["in_ten_color"] = ten_color
        elif component == 'out':
            left_x = self.trapezoid_upper_corners.get("out_left_x")
            left_y = self.trapezoid_upper_corners.get("out_left_y")
            right_x = self.trapezoid_upper_corners.get("out_right_x")
            right_y = self.trapezoid_upper_corners.get("out_right_y")
            ten_color = "#6B6B7A"
            self.component_settings["out_ten_color"] = ten_color
        
        # determine left x-coord start position for tendril
        x_range = right_x - left_x - diameter
        if position == 'left':
            tendril_left_x = left_x + x_range/4
        elif position == 'middle':
            tendril_left_x = left_x + x_range/2
        elif position == 'right':
            tendril_left_x = left_x + 3*x_range/4
        else:
            print "Error in draw_tendril: 'position' must be 'left', 'middle', or 'right'"
            return

        # slope of top leg
        slope = ((left_y - right_y) / (left_x - right_x))

        tendril_connect_left_y = left_y + slope * (tendril_left_x - left_x)

        tendril_right_x = tendril_left_x + diameter
        tendril_connect_right_y = left_y + slope * (tendril_right_x - left_x)

        # we don't want to go over the border (1.0) and we still need to close the tendril
        tendril_max_length = max_height - tendril_connect_left_y
        if tendril_max_length < 0:
            print "Error in draw_tendril: Can't draw Tendril here"
            return
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
            (tendril_left_x, tendril_connect_left_y),               # left side of tendril connected to trapezoid
            (control_point_low_left_x, control_point_low_y),        # curve tendril into one direction
            (tendril_left_x, tendril_mid_y),                        # middle of tendril
            (control_point_high_left_x, control_point_high_y),      # curve into other direction
            (tendril_left_x, tendril_top_left_y),                   # reached top
            # build round top:
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
        patch = patches.PathPatch(path, joinstyle='bevel', facecolor=ten_color, edgecolor='#70707D', lw=1.5, zorder=0.4)
        return patch

    def tube_nodes_as_chain(self, graph):
        out_nodes = graph.bow_tie_nodes[2]
        tube_nodes = graph.bow_tie_nodes[5]
        if len(tube_nodes) == 0:
            return []
        chain = []
        vprop_sp_tube = graph.new_vertex_property("int")

        # find shortest path to out component for every element in tube
        for out_node in out_nodes:
            for tube_node in tube_nodes:
                vlist, elist = gt.shortest_path(graph, graph.vertex(tube_node), graph.vertex(out_node))
                if (len(elist) < vprop_sp_tube[graph.vertex(tube_node)] or vprop_sp_tube[graph.vertex(tube_node)] is 0) and len(elist):
                    vprop_sp_tube[graph.vertex(tube_node)] = len(elist)

        # number of sections, equals the edges of the longest path
        tube_sections = vprop_sp_tube.a.max()
        for i in range(0, tube_sections):
            for tube_node in tube_nodes:
                if vprop_sp_tube[graph.vertex(tube_node)]-1 == i:
                    chain.append(tube_node)
        chain.reverse()
        return chain

    def tube_node_positions(self, graph, vprop_positions):
        upper_left_x = self.tube_key_points.get("upper_left_x")
        upper_left_y = self.tube_key_points.get("upper_left_y")
        upper_right_x = self.tube_key_points.get("upper_right_x")
        upper_right_y = self.tube_key_points.get("upper_right_y")
        upper_bezier_x = self.tube_key_points.get("upper_bezier_x")
        upper_bezier_y = self.tube_key_points.get("upper_bezier_y")
        diameter = self.component_settings.get("tube_diameter")

        chain = self.tube_nodes_as_chain(graph)
        x_range = upper_right_x - upper_left_x + diameter
        x_step = x_range/(len(chain)+1)
        upper_left_slope = (upper_left_y - upper_bezier_y) / (upper_left_x - upper_bezier_x)
        upper_right_slope = (upper_bezier_y - upper_right_y) / (upper_bezier_x - upper_right_x)

        """
        self.draw_point_patch(upper_left_x, upper_left_y)
        self.draw_point_patch(upper_right_x, upper_right_y)
        self.draw_point_patch(upper_bezier_x, upper_bezier_y)
        self.draw_point_patch(self.tube_key_points.get("lower_bezier_x"), self.tube_key_points.get("lower_bezier_y"))
        self.draw_point_patch(self.tube_key_points.get("lower_left_x"), self.tube_key_points.get("lower_left_y"))
        self.draw_point_patch(self.tube_key_points.get("lower_right_x"), self.tube_key_points.get("lower_right_y"))
        """

        last_y = upper_left_y - diameter/4
        first_up = True
        for i, node in enumerate(chain):
            x_pos = upper_left_x-diameter/2 + (i+1)*x_step
            if i+1 < (len(chain)/2):
                y_pos = last_y + upper_left_slope*x_step
            elif i >= (len(chain)/2):
                if not len(chain)%2 and first_up:
                    y_pos = last_y
                    first_up = False
                else:
                    y_pos = last_y + upper_right_slope*x_step
            else:
                y_pos = last_y + upper_left_slope*x_step
            vprop_positions[graph.vertex(node)] = np.array([x_pos, 1-y_pos])
            differencia = last_y-y_pos
            last_y = y_pos
            #self.draw_point_patch(x_pos, y_pos)

    def draw_tube(self, graph, diameter=0.075):
        # get trapezoid coordinates (but use the bottom side of trapezoid -> '1-')
        in_left_x = self.trapezoid_upper_corners.get("in_left_x")
        in_left_y = 1-self.trapezoid_upper_corners.get("in_left_y")
        in_right_x = self.trapezoid_upper_corners.get("in_right_x")
        in_right_y = 1-self.trapezoid_upper_corners.get("in_right_y")
        
        out_left_x = self.trapezoid_upper_corners.get("out_left_x")
        out_left_y = 1-self.trapezoid_upper_corners.get("out_left_y")
        out_right_x = self.trapezoid_upper_corners.get("out_right_x")
        out_right_y = 1-self.trapezoid_upper_corners.get("out_right_y")
        
        slope_in = ((in_left_y - in_right_y) / (in_left_x - in_right_x))
        slope_out = ((out_left_y - out_right_y) / (out_left_x - out_right_x))

        # path starts/ends in middle of trapezoids arms (trapezoids have different arm lengths)
        line1_start_x = in_left_x + (in_right_x - in_left_x)/2 + diameter/2
        line1_start_y = in_left_y + slope_in * (line1_start_x - in_left_x)

        line1_end_x = out_left_x + (out_right_x - out_left_x)/2 - diameter/2
        line1_end_y = out_left_y + slope_out * (line1_end_x - out_left_x)

        # calculate bézier control point
        y_diff = max((in_right_y - line1_start_y), (out_left_y - line1_end_y))
        control_point_one = (center_coordinate, (center_coordinate - self.scc_circle_radius - y_diff))

        control_point_two = (center_coordinate, control_point_one[1]-diameter)

        # calculate second line between trapezoids, this time we start at right trapezoid (path leads back)
        line2_start_x = line1_end_x + diameter
        line2_start_y = line1_end_y + slope_out * (line2_start_x - line1_end_x)
        line2_end_x = line1_start_x - diameter
        line2_end_y = line1_start_y - slope_in * (line1_start_x - line2_end_x)
        verts = [
            (line1_start_x, line1_start_y), # starts at in_trapezoid
            control_point_one,                  # move below scc circle
            (line1_end_x, line1_end_y),   # ends at out trapezoid
            (line2_start_x, line2_start_y), # starts at out trapezoid (lower)
            control_point_two,                  # moves below control point 1
            (line2_end_x, line2_end_y),   # ends at in trapezoid (lower than line1_start)
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
        tube_color = "#E6E6E8"
        patch = patches.PathPatch(path, facecolor=tube_color, edgecolor='#70707D', lw=1.5, zorder=0.4)

        # store settings for further use
        self.component_settings["tube_color"] = tube_color
        self.component_settings["tube_diameter"] = diameter

        self.tube_key_points["upper_left_x"] = line1_start_x
        self.tube_key_points["upper_left_y"] = line1_start_y
        self.tube_key_points["upper_right_x"] = line1_end_x
        self.tube_key_points["upper_right_y"] = line1_end_y
        self.tube_key_points["upper_bezier_x"] = control_point_one[0]
        self.tube_key_points["upper_bezier_y"] = control_point_one[1]

        self.tube_key_points["lower_left_x"] = line2_end_x
        self.tube_key_points["lower_left_y"] = line2_end_y
        self.tube_key_points["lower_right_x"] = line2_start_x
        self.tube_key_points["lower_right_y"] = line2_start_y
        self.tube_key_points["lower_bezier_x"] = control_point_two[0]
        self.tube_key_points["lower_bezier_y"] = control_point_two[1]
        return patch

    def draw_other(self, graph):
        other_color = "#DEBD5C"
        self.component_settings["other_color"] = other_color
        # TODO
        return 0

    def draw_orientation_lines(self):
        self.draw_vertical_line(0.25)
        self.draw_vertical_line(0.50)
        self.draw_vertical_line(0.75)
        self.draw_horizontal_line(0.25)
        self.draw_horizontal_line(0.50)
        self.draw_horizontal_line(0.75)

    def draw_vertical_line(self, x, y_padding=0):
        plt.plot((x, x), (y_padding, 1-y_padding), 'k--')

    def draw_section_lines(self):
        for line in self.sectionLines:
            self.draw_vertical_line(line[0], line[1])

    def draw_point_patch(self, x, y):
        rectangle = patches.Rectangle(xy=(x, y), height=.01, width=.01, angle=0, color='g', fill='True', zorder=1)
        plt.gca().add_patch(rectangle)

    def show_component_node_legend(self, graph, plt):
        patch_handles = []
        if len(graph.bow_tie_nodes[0]):
            inc_grouped_string = "INC: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[0]))
            in_color = self.component_settings.get("in_color")
            inc_patch = patches.Patch(label=inc_grouped_string, color=in_color)
            patch_handles.append(inc_patch)
        if len(graph.bow_tie_nodes[1]):
            scc_grouped_string = "SCC: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[1]))
            scc_color = self.component_settings.get("scc_color")
            scc_patch = patches.Patch(label=scc_grouped_string, color=scc_color)
            patch_handles.append(scc_patch)
        if len(graph.bow_tie_nodes[2]):
            out_grouped_string = "OUT: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[2]))
            out_color = self.component_settings.get("out_color")
            out_patch = patches.Patch(label=out_grouped_string, color=out_color)
            patch_handles.append(out_patch)
        if len(graph.bow_tie_nodes[3]):
            in_tendril_grouped_string = "I_T: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[3]))
            in_ten_color = self.component_settings.get("in_ten_color")
            in_tendril_patch = patches.Patch(label=in_tendril_grouped_string, color=in_ten_color)
            patch_handles.append(in_tendril_patch)
        if len(graph.bow_tie_nodes[4]):
            out_tendril_grouped_string = "O_T: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[4]))
            out_ten_color = self.component_settings.get("out_ten_color")
            out_tendril_patch = patches.Patch(label=out_tendril_grouped_string, color=out_ten_color)
            patch_handles.append(out_tendril_patch)
        if len(graph.bow_tie_nodes[5]):
            tube_grouped_string = "TUB: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[5]))
            tube_color = self.component_settings.get("tube_color")
            tube_patch = patches.Patch(label=tube_grouped_string, color=tube_color)
            patch_handles.append(tube_patch)
        if len(graph.bow_tie_nodes[6]):
            other_grouped_string = "OTH: \t" + self.group_numbers(sorted(graph.bow_tie_nodes[6]))
            other_color = self.component_settings.get("other_color")
            other_patch = patches.Patch(label=other_grouped_string, color=other_color)
            patch_handles.append(other_patch)
        plt.legend(loc=2, frameon=False, borderpad=0, borderaxespad=0, handles=patch_handles)

    def show_component_percent_legend(self):

        in_left_x = self.trapezoid_upper_corners.get("in_left_x")
        in_right_x = self.trapezoid_upper_corners.get("in_right_x")
        out_left_x = self.trapezoid_upper_corners.get("out_left_x")
        out_right_x = self.trapezoid_upper_corners.get("out_right_x")

        in_color = self.component_settings.get("in_color")
        scc_color = self.component_settings.get("scc_color")
        out_color = self.component_settings.get("out_color")

        in_c = (self.component_settings.get("in_percentage"))
        scc_c = (self.component_settings.get("scc_percentage"))
        out_c = (self.component_settings.get("out_percentage"))

        # create Percentage Labels below components if component > 0%
        # font size depending on component sizes
        max_label_font_size = 60
        min_label_font_size = 10
        font_size_range = max_label_font_size - min_label_font_size
        
        # see that percentage adds up to 100
        sum_percentage = 0
        largest_component = ""
        if in_c:
            in_c = int(round(in_c * 100))
            sum_percentage += in_c
            largest_component = "in"
        if scc_c:
            scc_c = int(round(scc_c * 100))
            sum_percentage += scc_c
            if scc_c > in_c:
                largest_component = "scc"
        if out_c:
            out_c = int(round(out_c * 100))
            sum_percentage += out_c
            if out_c > scc_c and out_c > in_c:
                largest_component = "out"

        if sum_percentage == 99:
            if largest_component == 'in':
                in_c += 1
            elif largest_component == 'out':
                out_c += 1
            elif largest_component == 'scc':
                scc_c += 1
        elif sum_percentage == 101:
            if largest_component == 'in':
                in_c -= 1
            elif largest_component == 'out':
                out_c -= 1
            elif largest_component == 'scc':
                scc_c -= 1

        # Percentage labels at the bottom
        # IN Component Label
        if in_c:
            x_range = in_right_x - in_left_x
            in_label_x_coord = in_left_x + x_range / 2
            in_fontsize = min_label_font_size + (font_size_range * in_c/100)
            plt.text(in_label_x_coord, 0.02, str(in_c) + "%", fontsize=in_fontsize, color=in_color)
        # SCC Component Label
        if scc_c:
            scc_fontsize = min_label_font_size + (font_size_range * scc_c/100)
            plt.text(center_coordinate, 0.02, str(scc_c) + "%", fontsize=scc_fontsize, color=scc_color)

        # OUT Component Label
        if out_c:
            x_range = out_right_x - out_left_x
            out_label_x_coord = out_left_x + x_range / 2
            out_fontsize = min_label_font_size + (font_size_range * out_c/100)
            plt.text(out_label_x_coord, 0.02, str(out_c) + "%", fontsize=out_fontsize, color=out_color)

    def show_component_size_legend(self):
        in_left_x = self.trapezoid_upper_corners.get("in_left_x")
        in_left_y = self.trapezoid_upper_corners.get("in_left_y")
        in_right_x = self.trapezoid_upper_corners.get("in_right_x")
        in_right_y = self.trapezoid_upper_corners.get("in_right_y")
        
        out_left_x = self.trapezoid_upper_corners.get("out_left_x")
        out_left_y = self.trapezoid_upper_corners.get("out_left_y")
        out_right_x = self.trapezoid_upper_corners.get("out_right_x")
        out_right_y = self.trapezoid_upper_corners.get("out_right_y")

        in_c = self.component_settings.get("in_percentage")
        scc_c = self.component_settings.get("scc_percentage")
        out_c = self.component_settings.get("out_percentage")

        scc_area = np.power(self.scc_circle_radius, 2)*np.pi
        # trapezoid area: A = (a+c)/2 * h
        if in_left_y > 0 and in_right_y > 0:
            in_area = self.trapezoid_without_circle_segment_area(in_left_x, in_left_y, in_right_x, in_right_y, 'in')
        if out_left_y > 0 and out_right_y > 0:
            # trapezoid area: A = (a+c)/2 * h
            a = (out_left_y - (1 - out_left_y))
            c = (out_right_y - (1 - out_right_y))
            h = out_right_x - out_left_x
            out_area = (a+c)/2 * h
            # calculate circular segment
            segment_height = (center_coordinate + self.scc_circle_radius) - out_left_x
            center_angle = 2 * atan(a/(2*(self.scc_circle_radius-segment_height)))
            scc_overlap_area = (np.power(self.scc_circle_radius, 2)/2) * (center_angle - sin(center_angle))
            out_area -= scc_overlap_area

    def trapezoid_without_circle_segment_area(self, left_x, left_y_top, right_x, right_y_top, component):
        if component == 'in':
            a = (right_y_top - (1 - right_y_top))
            c = (left_y_top - (1 - left_y_top))
            segment_height = right_x - (center_coordinate - self.scc_circle_radius)
        elif component == 'out':
            c = (right_y_top - (1 - right_y_top))
            a = (left_y_top - (1 - left_y_top))
            segment_height = (center_coordinate + self.scc_circle_radius) - left_x
        else:
            return
        h = right_x - left_x
        trapezoid_area = (a+c)/2 * h
        # calculate circular segment area
        center_angle = 2 * atan(a/(2*(self.scc_circle_radius-segment_height)))
        scc_overlap_area = (np.power(self.scc_circle_radius, 2)/2) * (center_angle - sin(center_angle))
        trapezoid_area -= scc_overlap_area
        return trapezoid_area

    def circle_segment_area(self, segment_height, chord):
        center_angle = 2 * atan(chord/(2*(self.scc_circle_radius-segment_height)))
        segment_area = (np.power(self.scc_circle_radius, 2)/2) * (center_angle - sin(center_angle))
        return segment_area

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

        grouped_items = ""
        first = True
        for group in groups:
            # semicolons only between groups, not in front of first, not behind last
            if first:
                first = False
            else:
                grouped_items += ", "
            if len(group) > 2:
                grouped_items += str(group[0]) + "-" + str(group[-1])
            elif len(group) == 2:
                grouped_items += str(group[0]) + ", " + str(group[-1])
            else:
                grouped_items += str(group[0])
        return grouped_items

"""
Main: Create one ugly random graph
"""
if __name__ == '__main__':
    # create one random graph
    graphs = []
    gc = GraphCollection('gnp')
    g = Graph()
    nodes = 25
    g.add_vertex(nodes)
    for i in range(0, 2*nodes):
        g.add_edge(g.vertex(random.randint(0, nodes-1)), g.vertex(random.randint(0, nodes-1)))
    gc.append(g)
    graphs.append(gc)

    # compute statistics
    for g in graphs:
        g.compute()

    # plot
    P = Plotting(graphs)
    P.plot_bowtie("ugly_random_graph", show_legends=False, show_sections=True, save_bg_file=False, save_graph_file=False,
                  only_background=False)
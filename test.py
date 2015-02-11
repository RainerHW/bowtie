# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import unittest
#import generators
import copy
import networkx as nx
import time as time

from main import Graph
from main import Plotting
from main import GraphCollection
"""
Class for Testing the Graph and its statistics
[Created by Daniel Lamprecht]
"""
class TestGraph(unittest.TestCase):
    def test_bowtie(self):
        graph = Graph()
        vList = graph.add_vertex(13)

        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(3), graph.vertex(1))

        graph.add_edge(graph.vertex(1), graph.vertex(0))

        graph.add_edge(graph.vertex(3), graph.vertex(4))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(6))
        graph.add_edge(graph.vertex(6), graph.vertex(4))

        graph.add_edge(graph.vertex(6), graph.vertex(7))

        graph.add_edge(graph.vertex(3), graph.vertex(9))
        graph.add_edge(graph.vertex(9), graph.vertex(10))
        graph.add_edge(graph.vertex(10), graph.vertex(11))
        graph.add_edge(graph.vertex(11), graph.vertex(7))

        graph.add_edge(graph.vertex(8), graph.vertex(7))
        graph.add_edge(graph.vertex(8), graph.vertex(12))

        graph.stats()
        self.assertEqual(graph.bow_tie,
                [
                300/len(list(graph.vertices())),
                300/len(list(graph.vertices())),
                100/len(list(graph.vertices())),
                100/len(list(graph.vertices())),
                100/len(list(graph.vertices())),
                300/len(list(graph.vertices())),
                100/len(list(graph.vertices()))
                ]
                )

"""
Class for Testing: Plotting 
"""
class TestPlotBowtie(unittest.TestCase):
    def test_tiny(self):
        print("Plotting Test Start:\tTiny Graph")
        start_time = time.time()
        graph = Graph()
        vList = graph.add_vertex(8)

        # build scc
        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(0), graph.vertex(2))
        graph.add_edge(graph.vertex(0), graph.vertex(3))
        
        graph.add_edge(graph.vertex(1), graph.vertex(0))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(1), graph.vertex(3))
        
        graph.add_edge(graph.vertex(2), graph.vertex(0))
        graph.add_edge(graph.vertex(2), graph.vertex(1))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        
        graph.add_edge(graph.vertex(3), graph.vertex(0))
        graph.add_edge(graph.vertex(3), graph.vertex(1))
        graph.add_edge(graph.vertex(3), graph.vertex(2))

        # build in
        graph.add_edge(graph.vertex(4), graph.vertex(0))
        graph.add_edge(graph.vertex(5), graph.vertex(1))
        
        # build out
        graph.add_edge(graph.vertex(2), graph.vertex(6))
        graph.add_edge(graph.vertex(3), graph.vertex(7))

        self.plot_graph(graph, "tiny")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tTiny Graph\t(%.2fs)" %duration)

    def test_small(self):
        print("Plotting Test Start:\tSmall Graph")
        start_time = time.time()
        name = "small"
        graph = Graph()
        vList = graph.add_vertex(14)

        # build scc
        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(3), graph.vertex(4))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(0))

        # build in
        graph.add_edge(graph.vertex(9), graph.vertex(0))
        graph.add_edge(graph.vertex(9), graph.vertex(5))
        graph.add_edge(graph.vertex(11), graph.vertex(5))
        graph.add_edge(graph.vertex(10), graph.vertex(11))
        graph.add_edge(graph.vertex(11), graph.vertex(12))
        graph.add_edge(graph.vertex(12), graph.vertex(10))
        graph.add_edge(graph.vertex(13), graph.vertex(5))
        graph.add_edge(graph.vertex(13), graph.vertex(4))

        # build out
        graph.add_edge(graph.vertex(1), graph.vertex(6))
        graph.add_edge(graph.vertex(2), graph.vertex(7))
        graph.add_edge(graph.vertex(3), graph.vertex(8))

        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tSmall Graph\t(%.2fs)" %duration)

    def test_medium(self):
        print("Plotting Test Start:\tMedium Graph ")
        start_time = time.time()
        name = "medium"
        graph = Graph()
        vList = graph.add_vertex(20)

        # build in
        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(3), graph.vertex(1))
        graph.add_edge(graph.vertex(16), graph.vertex(1))

        graph.add_edge(graph.vertex(1), graph.vertex(0))

        # build scc
        graph.add_edge(graph.vertex(2), graph.vertex(5))
        graph.add_edge(graph.vertex(3), graph.vertex(4))
        graph.add_edge(graph.vertex(3), graph.vertex(6))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(6))
        graph.add_edge(graph.vertex(5), graph.vertex(7))
        graph.add_edge(graph.vertex(6), graph.vertex(4))
        graph.add_edge(graph.vertex(2), graph.vertex(13))
        graph.add_edge(graph.vertex(15), graph.vertex(6))
        graph.add_edge(graph.vertex(4), graph.vertex(15))
        graph.add_edge(graph.vertex(13), graph.vertex(15))
        graph.add_edge(graph.vertex(4), graph.vertex(13))

        # build out
        graph.add_edge(graph.vertex(6), graph.vertex(7))
        graph.add_edge(graph.vertex(13), graph.vertex(14))
        graph.add_edge(graph.vertex(14), graph.vertex(17))
        graph.add_edge(graph.vertex(14), graph.vertex(18))
        graph.add_edge(graph.vertex(17), graph.vertex(18))

        # build tube
        graph.add_edge(graph.vertex(3), graph.vertex(9))
        graph.add_edge(graph.vertex(9), graph.vertex(10))
        graph.add_edge(graph.vertex(10), graph.vertex(11))
        graph.add_edge(graph.vertex(11), graph.vertex(7))

        # Out-Tendril
        graph.add_edge(graph.vertex(8), graph.vertex(7))

        # Other
        graph.add_edge(graph.vertex(8), graph.vertex(12))

        # In-Tendril
        graph.add_edge(graph.vertex(16), graph.vertex(19))

        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tMedium Graph \t(%.2fs)" %duration)

    def test_tube(self):
        print("Plotting Test Start:\tTube ")
        start_time = time.time()
        name = "tube"
        graph = Graph()
        vList = graph.add_vertex(20)

        # build in
        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(3), graph.vertex(1))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(6), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(3))

        # build scc
        graph.add_edge(graph.vertex(3), graph.vertex(7))
        graph.add_edge(graph.vertex(2), graph.vertex(11))
        graph.add_edge(graph.vertex(7), graph.vertex(8))
        graph.add_edge(graph.vertex(8), graph.vertex(9))
        graph.add_edge(graph.vertex(9), graph.vertex(10))
        graph.add_edge(graph.vertex(10), graph.vertex(11))
        graph.add_edge(graph.vertex(11), graph.vertex(7))

        # build out
        graph.add_edge(graph.vertex(8), graph.vertex(12))
        graph.add_edge(graph.vertex(12), graph.vertex(13))
        graph.add_edge(graph.vertex(12), graph.vertex(14))
        graph.add_edge(graph.vertex(9), graph.vertex(15))
        graph.add_edge(graph.vertex(15), graph.vertex(16))

        # build tube
        graph.add_edge(graph.vertex(5), graph.vertex(17))
        graph.add_edge(graph.vertex(17), graph.vertex(18))
        graph.add_edge(graph.vertex(18), graph.vertex(19))
        graph.add_edge(graph.vertex(19), graph.vertex(16))

        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tTube \t(%.2fs)" %duration)

    def test_grow_in(self):
        print("Plotting Test Start:\tGrow IN")
        start_time = time.time()
        name = "growIn"
        graph = Graph()

        graphs = []
        # create graphs collection instance 
        # is needed to create plotting instance
        gc = GraphCollection(name)
        vList = graph.add_vertex(7)

        # SCC
        graph.add_edge(graph.vertex(0),graph.vertex(1))
        graph.add_edge(graph.vertex(1),graph.vertex(2))
        graph.add_edge(graph.vertex(2),graph.vertex(3))
        graph.add_edge(graph.vertex(3),graph.vertex(4))
        graph.add_edge(graph.vertex(4),graph.vertex(5))
        graph.add_edge(graph.vertex(5),graph.vertex(6))
        graph.add_edge(graph.vertex(6),graph.vertex(0))
        gc.append(copy.deepcopy(graph))

        print("Growing IN")
        for x in range (7, 17):
            new_vertex = graph.add_vertex()
            graph.add_edge(new_vertex, graph.vertex(1))
            gc.append(copy.deepcopy(graph))

        graphs.append(gc)
        print("Graphs Created! Starting the plotting")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow IN\t\t(%.2fs)" %duration)

    def test_grow_in_scc_out(self):
        print("Plotting Test Start:\tGrow")
        start_time = time.time()
        name = "grow_in_scc_out"
        graph = Graph()

        graphs = []
        # create graphs collection instance 
        gc = GraphCollection(name)

        graph.add_vertex(7)
        # SCC
        graph.add_edge(graph.vertex(0),graph.vertex(1))
        graph.add_edge(graph.vertex(1),graph.vertex(2))
        graph.add_edge(graph.vertex(2),graph.vertex(3))
        graph.add_edge(graph.vertex(3),graph.vertex(4))
        graph.add_edge(graph.vertex(4),graph.vertex(5))
        graph.add_edge(graph.vertex(5),graph.vertex(6))
        graph.add_edge(graph.vertex(6),graph.vertex(0))
        gc.append(copy.deepcopy(graph))

        print("Growing IN")
        for x in range (7, 27):
            new_vertex = graph.add_vertex()
            graph.add_edge(new_vertex, graph.vertex(1))
            gc.append(copy.deepcopy(graph))
        
        print("Growing OUT")
        for x in range (27, 47):
            new_vertex = graph.add_vertex()
            graph.add_edge(graph.vertex(2), new_vertex)
            gc.append(copy.deepcopy(graph))
        
        print("Growing SCC")
        for x in range (7, 27):
            graph.add_edge(graph.vertex(1), graph.vertex(x))
            gc.append(copy.deepcopy(graph))
        
            graph.add_edge(graph.vertex(20+x), graph.vertex(2))
            gc.append(copy.deepcopy(graph))

        graphs.append(gc)
        print("Graphs Created! Starting the plotting")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow\t\t(%.2fs)" %duration)

    def test_grow_shrink(self):
        print("Plotting Test Start:\tGrow & Shrink (GC)")
        start_time = time.time()
        name = "grow_shrink"
        graph = Graph()
        
        graphs = []
        # create graphs collection instance 
        gc = GraphCollection(name)

        graph.add_vertex(7)
        # SCC
        graph.add_edge(graph.vertex(0),graph.vertex(1))
        graph.add_edge(graph.vertex(1),graph.vertex(2))
        graph.add_edge(graph.vertex(2),graph.vertex(3))
        graph.add_edge(graph.vertex(3),graph.vertex(4))
        graph.add_edge(graph.vertex(4),graph.vertex(5))
        graph.add_edge(graph.vertex(5),graph.vertex(6))
        graph.add_edge(graph.vertex(6),graph.vertex(0))
        gc.append(copy.deepcopy(graph))

        in_edges = []
        out_edges = []
        scc_edges = []

        print("Growing IN")
        for x in range (7, 27):
            new_vertex = graph.add_vertex()
            in_edge = graph.add_edge(new_vertex, graph.vertex(1))
            in_edges.append(in_edge)
            gc.append(copy.deepcopy(graph))
        
        print("Growing OUT")
        for x in range (27, 47):
            new_vertex = graph.add_vertex()
            out_edge = graph.add_edge(graph.vertex(2), new_vertex)
            out_edges.append(out_edge)
            gc.append(copy.deepcopy(graph))
        
        print("Growing SCC")
        for x in range (7, 27):
            scc_edge = graph.add_edge(graph.vertex(1), graph.vertex(x))
            scc_edges.append(scc_edge)
            gc.append(copy.deepcopy(graph))
        
            scc_edge = graph.add_edge(graph.vertex(20+x), graph.vertex(2))
            scc_edges.append(scc_edge)
            gc.append(copy.deepcopy(graph))

        print("Shrinking SCC")
        for edge in scc_edges:
            graph.remove_edge(edge)
            gc.append(copy.deepcopy(graph))

        print("Shrinking OUT")
        for edge in out_edges:
            graph.remove_edge(edge)
            gc.append(copy.deepcopy(graph))

        print("Shrinking IN")
        for edge in in_edges:
            graph.remove_edge(edge)
            gc.append(copy.deepcopy(graph))

        graphs.append(gc)
        print("Plotting Graphs")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow & Shrink (deep copy)\t(%.2fs)" %duration)
    
    def test_10nodes_in_scc_out(self):
        print("Plotting Test Start:\t10 Nodes ")
        start_time = time.time()
        name = "10_nodes"
        graph = Graph()
        vList = graph.add_vertex(10)

        # IN-Layers
        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(1), graph.vertex(3))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(3), graph.vertex(4))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(6))
        graph.add_edge(graph.vertex(6), graph.vertex(7))
        graph.add_edge(graph.vertex(7), graph.vertex(5))
        graph.add_edge(graph.vertex(7), graph.vertex(8))
        graph.add_edge(graph.vertex(7), graph.vertex(9))
        graph.add_edge(graph.vertex(8), graph.vertex(6))
        graph.add_edge(graph.vertex(8), graph.vertex(9))

        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\t10 Nodes \t(%.2fs)" %duration)

    def test_20nodes_in_scc_out(self):
        print("Plotting Test Start:\t20 Nodes ")
        start_time = time.time()
        name = "20_nodes"
        graph = Graph()
        vList = graph.add_vertex(20)

        # IN-Layers
        graph.add_edge(graph.vertex(0), graph.vertex(2))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(2), graph.vertex(4))

        graph.add_edge(graph.vertex(3), graph.vertex(5))
        graph.add_edge(graph.vertex(3), graph.vertex(6))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(7))
        graph.add_edge(graph.vertex(6), graph.vertex(8))
        graph.add_edge(graph.vertex(7), graph.vertex(6))
        graph.add_edge(graph.vertex(7), graph.vertex(9))
        graph.add_edge(graph.vertex(7), graph.vertex(10))
        graph.add_edge(graph.vertex(7), graph.vertex(12))
        graph.add_edge(graph.vertex(8), graph.vertex(5))
        graph.add_edge(graph.vertex(8), graph.vertex(9))
        graph.add_edge(graph.vertex(9), graph.vertex(10))
        graph.add_edge(graph.vertex(10), graph.vertex(11))
        graph.add_edge(graph.vertex(11), graph.vertex(8))
        graph.add_edge(graph.vertex(12), graph.vertex(10))

        graph.add_edge(graph.vertex(10), graph.vertex(13))
        graph.add_edge(graph.vertex(10), graph.vertex(14))
        graph.add_edge(graph.vertex(10), graph.vertex(15))
        graph.add_edge(graph.vertex(11), graph.vertex(17))
        graph.add_edge(graph.vertex(11), graph.vertex(18))
        graph.add_edge(graph.vertex(11), graph.vertex(14))
        graph.add_edge(graph.vertex(11), graph.vertex(15))
        graph.add_edge(graph.vertex(11), graph.vertex(16))
        graph.add_edge(graph.vertex(12), graph.vertex(18))
        graph.add_edge(graph.vertex(18), graph.vertex(19))

        
        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\t20 Nodes \t(%.2fs)" %duration)

    def test_30nodes_in_scc_out(self):
        print("Plotting Test Start:\t30 Nodes ")
        start_time = time.time()
        name = "30_nodes"
        graph = Graph()
        vList = graph.add_vertex(30)

        # IN-Layers
        graph.add_edge(graph.vertex(0), graph.vertex(3))
        graph.add_edge(graph.vertex(1), graph.vertex(3))
        graph.add_edge(graph.vertex(2), graph.vertex(4))
        graph.add_edge(graph.vertex(3), graph.vertex(5))
        graph.add_edge(graph.vertex(4), graph.vertex(3))
        graph.add_edge(graph.vertex(4), graph.vertex(6))
        graph.add_edge(graph.vertex(5), graph.vertex(6))
        graph.add_edge(graph.vertex(5), graph.vertex(7))
        graph.add_edge(graph.vertex(6), graph.vertex(7))
        graph.add_edge(graph.vertex(6), graph.vertex(8))

        # scc
        # in from scc (8,9,10,11)
        graph.add_edge(graph.vertex(7), graph.vertex(9))
        graph.add_edge(graph.vertex(7), graph.vertex(10))
        graph.add_edge(graph.vertex(7), graph.vertex(11))
        graph.add_edge(graph.vertex(8), graph.vertex(9))
        graph.add_edge(graph.vertex(9), graph.vertex(10))
        graph.add_edge(graph.vertex(9), graph.vertex(11))
        graph.add_edge(graph.vertex(10), graph.vertex(11)) 
        graph.add_edge(graph.vertex(10), graph.vertex(14))
        graph.add_edge(graph.vertex(11), graph.vertex(8))
        graph.add_edge(graph.vertex(11), graph.vertex(12))
        graph.add_edge(graph.vertex(12), graph.vertex(13))
        graph.add_edge(graph.vertex(13), graph.vertex(11))
        graph.add_edge(graph.vertex(13), graph.vertex(16))
        graph.add_edge(graph.vertex(14), graph.vertex(9))
        graph.add_edge(graph.vertex(14), graph.vertex(15))
        graph.add_edge(graph.vertex(14), graph.vertex(17))
        graph.add_edge(graph.vertex(15), graph.vertex(18))
        graph.add_edge(graph.vertex(15), graph.vertex(10))
        graph.add_edge(graph.vertex(16), graph.vertex(12))
        graph.add_edge(graph.vertex(16), graph.vertex(17))
        graph.add_edge(graph.vertex(11), graph.vertex(18))

        # OUT-Layers
        graph.add_edge(graph.vertex(18), graph.vertex(19))
        graph.add_edge(graph.vertex(18), graph.vertex(20))
        graph.add_edge(graph.vertex(18), graph.vertex(21))
        graph.add_edge(graph.vertex(17), graph.vertex(20))
        graph.add_edge(graph.vertex(17), graph.vertex(22))
        graph.add_edge(graph.vertex(17), graph.vertex(23))
        
        graph.add_edge(graph.vertex(19), graph.vertex(24))
        graph.add_edge(graph.vertex(20), graph.vertex(24))
        graph.add_edge(graph.vertex(20), graph.vertex(25))
        graph.add_edge(graph.vertex(20), graph.vertex(26))
        graph.add_edge(graph.vertex(21), graph.vertex(26))
        graph.add_edge(graph.vertex(22), graph.vertex(26))
        graph.add_edge(graph.vertex(22), graph.vertex(27))
        graph.add_edge(graph.vertex(23), graph.vertex(27))
        graph.add_edge(graph.vertex(24), graph.vertex(28))
        graph.add_edge(graph.vertex(27), graph.vertex(28))
        graph.add_edge(graph.vertex(27), graph.vertex(29))
        graph.add_edge(graph.vertex(28), graph.vertex(23))

        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\t30 Nodes \t(%.2fs)" %duration)

    def test_40nodes_in_scc_out(self):
        print("Plotting Test Start:\t40 Nodes ")
        start_time = time.time()
        name = "40_nodes"
        graph = Graph()
        vList = graph.add_vertex(40)

        # 1st and 2nd layer IN
        graph.add_edge(graph.vertex(0), graph.vertex(6))
        graph.add_edge(graph.vertex(1), graph.vertex(6))
        graph.add_edge(graph.vertex(2), graph.vertex(6))
        graph.add_edge(graph.vertex(3), graph.vertex(6))
        graph.add_edge(graph.vertex(4), graph.vertex(6))

        # 3rd layer IN
        graph.add_edge(graph.vertex(6), graph.vertex(7))
        graph.add_edge(graph.vertex(6), graph.vertex(8))
        graph.add_edge(graph.vertex(6), graph.vertex(9))
        graph.add_edge(graph.vertex(6), graph.vertex(10))

        # 4th layer IN
        graph.add_edge(graph.vertex(7), graph.vertex(11))
        graph.add_edge(graph.vertex(8), graph.vertex(12))
        graph.add_edge(graph.vertex(9), graph.vertex(11))
        graph.add_edge(graph.vertex(10), graph.vertex(12))

        # 5th layer IN (13, 14, 15, 16, 17)
        graph.add_edge(graph.vertex(11), graph.vertex(13))
        graph.add_edge(graph.vertex(11), graph.vertex(14))
        graph.add_edge(graph.vertex(11), graph.vertex(15))
        graph.add_edge(graph.vertex(12), graph.vertex(16))
        graph.add_edge(graph.vertex(12), graph.vertex(17))
        
        # build SCC
        # scc nodes from in (18, 19, 20)
        graph.add_edge(graph.vertex(13), graph.vertex(18))
        graph.add_edge(graph.vertex(14), graph.vertex(19))
        graph.add_edge(graph.vertex(15), graph.vertex(18))
        graph.add_edge(graph.vertex(16), graph.vertex(19))
        graph.add_edge(graph.vertex(17), graph.vertex(20))

        # scc nodes from in and to out (21, 22, 23, 24)
        graph.add_edge(graph.vertex(14), graph.vertex(21))
        graph.add_edge(graph.vertex(15), graph.vertex(22))
        graph.add_edge(graph.vertex(16), graph.vertex(23))
        graph.add_edge(graph.vertex(17), graph.vertex(24))
        graph.add_edge(graph.vertex(21), graph.vertex(31))
        graph.add_edge(graph.vertex(22), graph.vertex(31))
        graph.add_edge(graph.vertex(23), graph.vertex(30))
        graph.add_edge(graph.vertex(24), graph.vertex(29))

        # Internal SCC (25, 26)
        graph.add_edge(graph.vertex(18), graph.vertex(24))
        graph.add_edge(graph.vertex(18), graph.vertex(23))
        graph.add_edge(graph.vertex(19), graph.vertex(23))
        graph.add_edge(graph.vertex(19), graph.vertex(22))
        graph.add_edge(graph.vertex(18), graph.vertex(25))
        graph.add_edge(graph.vertex(20), graph.vertex(26))
        graph.add_edge(graph.vertex(21), graph.vertex(27))
        graph.add_edge(graph.vertex(22), graph.vertex(27))
        graph.add_edge(graph.vertex(24), graph.vertex(28))
        graph.add_edge(graph.vertex(25), graph.vertex(21))
        graph.add_edge(graph.vertex(27), graph.vertex(28))
        graph.add_edge(graph.vertex(28), graph.vertex(20))

        # back links (to ensure scc)
        graph.add_edge(graph.vertex(25), graph.vertex(19))
        graph.add_edge(graph.vertex(25), graph.vertex(20))
        graph.add_edge(graph.vertex(26), graph.vertex(18))
        graph.add_edge(graph.vertex(21), graph.vertex(22))
        graph.add_edge(graph.vertex(23), graph.vertex(24))

        # scc nodes to out (27, 28)
        graph.add_edge(graph.vertex(27), graph.vertex(29))
        graph.add_edge(graph.vertex(28), graph.vertex(29))
        graph.add_edge(graph.vertex(28), graph.vertex(30))
        graph.add_edge(graph.vertex(28), graph.vertex(31))

        # multiple out layers
        graph.add_edge(graph.vertex(31), graph.vertex(32))
        graph.add_edge(graph.vertex(32), graph.vertex(33))
        graph.add_edge(graph.vertex(32), graph.vertex(34))
        graph.add_edge(graph.vertex(32), graph.vertex(35))
        graph.add_edge(graph.vertex(34), graph.vertex(36))
        graph.add_edge(graph.vertex(35), graph.vertex(37))
        graph.add_edge(graph.vertex(37), graph.vertex(38))
        graph.add_edge(graph.vertex(37), graph.vertex(39))

        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\t40 Nodes \t(%.2fs)" %duration)

    def test_big_in(self):
        print("Plotting Test Start:\tBig In ")
        start_time = time.time()
        name = "big_in"
        graph = Graph()
        vList = graph.add_vertex(22)

        # IN-Layers
        graph.add_edge(graph.vertex(0), graph.vertex(7))
        graph.add_edge(graph.vertex(1), graph.vertex(8))
        graph.add_edge(graph.vertex(2), graph.vertex(7))
        graph.add_edge(graph.vertex(3), graph.vertex(8))
        graph.add_edge(graph.vertex(4), graph.vertex(7))
        graph.add_edge(graph.vertex(5), graph.vertex(8))
        graph.add_edge(graph.vertex(6), graph.vertex(7))

        graph.add_edge(graph.vertex(7), graph.vertex(9))
        graph.add_edge(graph.vertex(7), graph.vertex(10))
        graph.add_edge(graph.vertex(7), graph.vertex(11))
        graph.add_edge(graph.vertex(8), graph.vertex(12))
        graph.add_edge(graph.vertex(8), graph.vertex(13))
        graph.add_edge(graph.vertex(8), graph.vertex(14))

        graph.add_edge(graph.vertex(9), graph.vertex(15))
        graph.add_edge(graph.vertex(10), graph.vertex(15))
        graph.add_edge(graph.vertex(11), graph.vertex(16))
        graph.add_edge(graph.vertex(12), graph.vertex(16))
        graph.add_edge(graph.vertex(13), graph.vertex(17))
        graph.add_edge(graph.vertex(14), graph.vertex(17))

        graph.add_edge(graph.vertex(15), graph.vertex(18))
        graph.add_edge(graph.vertex(16), graph.vertex(18))
        graph.add_edge(graph.vertex(17), graph.vertex(18))

        graph.add_edge(graph.vertex(18), graph.vertex(19))
        graph.add_edge(graph.vertex(19), graph.vertex(20))
        graph.add_edge(graph.vertex(20), graph.vertex(21))
        graph.add_edge(graph.vertex(21), graph.vertex(19))

                
        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig In \t\t(%.2fs)" %duration)

    def test_big_scc(self):
        print("Plotting Test Start:\tBig SCC")
        start_time = time.time()
        graph = Graph()
        graph.add_vertex(13)

        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(3))
        graph.add_edge(graph.vertex(3), graph.vertex(4))
        graph.add_edge(graph.vertex(4), graph.vertex(5))
        graph.add_edge(graph.vertex(5), graph.vertex(6))
        graph.add_edge(graph.vertex(6), graph.vertex(7))
        graph.add_edge(graph.vertex(7), graph.vertex(8))
        graph.add_edge(graph.vertex(8), graph.vertex(9))
        graph.add_edge(graph.vertex(9), graph.vertex(10))
        graph.add_edge(graph.vertex(10), graph.vertex(11))
        graph.add_edge(graph.vertex(11), graph.vertex(12))
        graph.add_edge(graph.vertex(12), graph.vertex(1))
        self.plot_graph(graph, "big_scc")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig SCC\t\t(%.2fs)" %duration)

    def test_big_out(self):
        print("Plotting Test Start:\tBig Out ")
        start_time = time.time()
        name = "big_out"
        graph = Graph()
        vList = graph.add_vertex(22)

        # IN-Layers
        graph.add_edge(graph.vertex(0), graph.vertex(1))
        graph.add_edge(graph.vertex(1), graph.vertex(2))
        graph.add_edge(graph.vertex(2), graph.vertex(0))
        graph.add_edge(graph.vertex(2), graph.vertex(3))

        graph.add_edge(graph.vertex(3), graph.vertex(4))
        graph.add_edge(graph.vertex(3), graph.vertex(5))
        graph.add_edge(graph.vertex(3), graph.vertex(6))

        graph.add_edge(graph.vertex(4), graph.vertex(7))
        graph.add_edge(graph.vertex(4), graph.vertex(8))
        graph.add_edge(graph.vertex(5), graph.vertex(9))
        graph.add_edge(graph.vertex(5), graph.vertex(10))
        graph.add_edge(graph.vertex(6), graph.vertex(11))
        graph.add_edge(graph.vertex(6), graph.vertex(12))

        graph.add_edge(graph.vertex(7), graph.vertex(13))
        graph.add_edge(graph.vertex(8), graph.vertex(13))
        graph.add_edge(graph.vertex(9), graph.vertex(13))
        graph.add_edge(graph.vertex(10), graph.vertex(14))
        graph.add_edge(graph.vertex(11), graph.vertex(14))
        graph.add_edge(graph.vertex(12), graph.vertex(14))

        graph.add_edge(graph.vertex(13), graph.vertex(15))
        graph.add_edge(graph.vertex(14), graph.vertex(16))
        graph.add_edge(graph.vertex(13), graph.vertex(17))
        graph.add_edge(graph.vertex(14), graph.vertex(18))
        graph.add_edge(graph.vertex(13), graph.vertex(19))
        graph.add_edge(graph.vertex(14), graph.vertex(20))
        graph.add_edge(graph.vertex(13), graph.vertex(21))

                
        self.plot_graph(graph, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig Out \t(%.2fs)" %duration)

    def plot_graph(self, graph, testcase):
        """
        Helper Method: Create the GC, Plotting object and call plot_bowtie
        Arguments:
        - testcase: the name of the TestCase
        - iteration: the iteration of the TestCase
        """

        graphs = []
        # create graphs collection instance 
        # is needed to create plotting instance
        gc = GraphCollection(testcase)
        gc.append(graph)
        graphs.append(gc)
        
        self.plot_graph_collection(graphs, testcase)
        
    def plot_graph_collection(self, graphs, testcase):
        #compute statistics for graphs
        for gc in graphs:
            gc.compute()
        # create Plotting instance and call the plotting method
        plotting = Plotting(graphs)
        plotting.plot_bowtie(testcase)

if __name__ == '__main__':
    def test_stats():
        # test bow tie stats (by daniel)
        t = TestGraph('test_bowtie')
        t.test_bowtie()

    def test_plotting():
        # test plotting
        p = TestPlotBowtie('test_tiny')

        # uncomment the test-cases you want to run
        #p.test_tiny()
        #p.test_small()
        p.test_medium()
        p.test_tube()
        #p.test_big_in()
        #p.test_big_scc()
        #p.test_big_out()
        #p.test_10nodes_in_scc_out()
        #p.test_20nodes_in_scc_out()
        #p.test_30nodes_in_scc_out()
        #p.test_40nodes_in_scc_out()
        #p.test_grow_in()
        #p.test_grow_in_scc_out()
        #p.test_grow_shrink()

    # run all stats and plotting test cases
    #test_stats()
    test_plotting()
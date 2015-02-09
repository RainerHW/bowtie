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
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)

        graph.add_edge(1, 0)

        graph.add_edge(3, 4)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        graph.add_edge(6, 4)

        graph.add_edge(6, 7)

        graph.add_edge(3, 9)
        graph.add_edge(9, 10)
        graph.add_edge(10, 11)
        graph.add_edge(11, 7)

        graph.add_edge(8, 7)
        graph.add_edge(8, 12)

        graph.stats()
        self.assertEqual(graph.bow_tie, [300/len(graph), 300/len(graph),
                                         100/len(graph), 100/len(graph),
                                         100/len(graph), 300/len(graph),
                                         100/len(graph)])

"""
Class for Testing: Plotting 
"""


class TestPlotting(unittest.TestCase):
    def test_bowtie_plot_tiny(self):
        print("Plotting Test Start:\tTiny Graph")
        start_time = time.time()
        graph = Graph()

        # build scc
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        
        graph.add_edge(2, 1)
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        
        graph.add_edge(3, 1)
        graph.add_edge(3, 2)
        graph.add_edge(3, 4)
        
        graph.add_edge(4, 1)
        graph.add_edge(4, 2)
        graph.add_edge(4, 3)

        # build in
        graph.add_edge(5, 1)
        graph.add_edge(6, 2)
        
        # build out
        graph.add_edge(3, 7)
        graph.add_edge(4, 8)

        self.plot_graph(graph, "tiny")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tTiny Graph\t(%.2fs)" %duration)

    def test_bowtie_plot_small(self):
        print("Plotting Test Start:\tSmall Graph")
        start_time = time.time()
        graph = Graph()

        # build scc
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        graph.add_edge(6, 1)

        # build in
        graph.add_edge(10, 1)
        graph.add_edge(10, 6)
        graph.add_edge(12, 6)
        graph.add_edge(11, 12)
        graph.add_edge(12, 13)
        graph.add_edge(13, 11)
        graph.add_edge(14, 6)
        graph.add_edge(14, 5)

        # build out
        graph.add_edge(2, 7)
        graph.add_edge(3, 8)
        graph.add_edge(4, 9)

        self.plot_graph(graph, "small")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tSmall Graph\t(%.2fs)" %duration)

    def test_bowtie_plot_medium(self):
        print("Plotting Test Start:\tMedium Graph")
        start_time = time.time()

        graph = Graph()
        # build in
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)
        graph.add_edge(16, 1)

        graph.add_edge(1, 0)

        # build scc
        graph.add_edge(2, 5)
        graph.add_edge(3, 4)
        graph.add_edge(3, 6)
        graph.add_edge(4, 5)
        graph.add_edge(5, 6)
        graph.add_edge(5, 7)
        graph.add_edge(6, 4)
        graph.add_edge(2, 13)
        graph.add_edge(15, 6)
        graph.add_edge(4, 15)
        graph.add_edge(13, 15)
        graph.add_edge(4, 13)

        # build out
        graph.add_edge(6, 7)
        graph.add_edge(13, 14)
        graph.add_edge(14, 17)
        graph.add_edge(14, 18)
        graph.add_edge(17, 18)

        # build tube
        graph.add_edge(3, 9)
        graph.add_edge(9, 10)
        graph.add_edge(10, 11)
        graph.add_edge(11, 7)

        graph.add_edge(8, 7)
        graph.add_edge(8, 12)

        graph.add_edge(16, 19)

        self.plot_graph(graph, "medium")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tMedium Graph\t(%.2fs)" %duration)

    def test_bowtie_plot_medium_gt(self):
        print("Plotting Test Start:\tMedium Graph (GT)")
        start_time = time.time()

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

        self.plot_graph(graph, "medium")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tMedium Graph (GT)\t(%.2fs)" %duration)

    def test_bowtie_plot_bigIN(self):
        """
        Creates a graph with a big IN-Component and a small SCC component
        """
        print("Plotting Test Start:\tBig IN")
        start_time = time.time()
        graph = Graph()
        graph.add_edge(1,2)
        graph.add_edge(3,2)
        graph.add_edge(4,2)
        graph.add_edge(5,2)
        graph.add_edge(6,2)
        graph.add_edge(7,2)
        graph.add_edge(8,2)
        graph.add_edge(9,2)
        graph.add_edge(10,2)
        graph.add_edge(11,2)
        graph.add_edge(12,2)
        graph.add_edge(13,2)
        self.plot_graph(graph, "bigIN")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig IN\t\t(%.2fs)" %duration)

    def test_bowtie_plot_bigSCC(self):
        print("Plotting Test Start:\tBig SCC")
        start_time = time.time()
        graph = Graph()
        graph.add_edge(1,2)
        graph.add_edge(2,3)
        graph.add_edge(3,4)
        graph.add_edge(4,5)
        graph.add_edge(5,6)
        graph.add_edge(6,7)
        graph.add_edge(7,8)
        graph.add_edge(8,9)
        graph.add_edge(9,10)
        graph.add_edge(10,11)
        graph.add_edge(11,12)
        graph.add_edge(12,13)
        graph.add_edge(13,2)
        self.plot_graph(graph, "bigSCC")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig SCC\t\t(%.2fs)" %duration)

    def test_bowtie_plot_growIN(self):
        print("Plotting Test Start:\tGrow IN")
        start_time = time.time()
        name = "growIn"
        graph = Graph()

        graphs = []
        # create graphs collection instance 
        # is needed to create plotting instance
        gc = GraphCollection(name)

        # SCC
        graph.add_edge(0,1)
        graph.add_edge(1,2)
        graph.add_edge(2,3)
        graph.add_edge(3,4)
        graph.add_edge(4,5)
        graph.add_edge(5,6)
        graph.add_edge(6,0)
        gc.append(copy.deepcopy(graph))

        print("Growing IN")
        for x in range (7, 17):
            graph.add_edge(x, 1)
            gc.append(copy.deepcopy(graph))

        graphs.append(gc)
        print("Graphs Created! Starting the plotting")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow IN\t\t(%.2fs)" %duration)

    def test_bowtie_plot_growComponents(self):
        print("Plotting Test Start:\tGrow")
        start_time = time.time()
        name = "growComponents"
        graph = Graph()
        iteration = 0

        graphs = []
        # create graphs collection instance 
        gc = GraphCollection(name)

        # SCC
        graph.add_edge(0,1)
        graph.add_edge(1,2)
        graph.add_edge(2,3)
        graph.add_edge(3,4)
        graph.add_edge(4,5)
        graph.add_edge(5,6)
        graph.add_edge(6,0)
        gc.append(copy.deepcopy(graph))

        print("Growing IN")
        for x in range (7, 17):
            graph.add_edge(x, 1)
            gc.append(copy.deepcopy(graph))

        print("Growing OUT")
        for x in range (17, 27):
            graph.add_edge(2, x)
            gc.append(copy.deepcopy(graph))
        
        print("Growing SCC")
        for x in range (7, 17):
            graph.add_edge(1, x)
            gc.append(copy.deepcopy(graph))
        
            graph.add_edge(x+10, 2)
            gc.append(copy.deepcopy(graph))

        graphs.append(gc)
        print("Graphs Created! Starting the plotting")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow\t\t(%.2fs)" %duration)

    def test_bowtie_plot_growAndShrinkComponents(self):
        print("Plotting Test Start:\tGrow & Shrink")
        start_time = time.time()
        name = "growAndShrinkComponents"
        graph = Graph()
        iteration = 0

        # SCC
        graph.add_edge(0,1)
        graph.add_edge(1,2)
        graph.add_edge(2,3)
        graph.add_edge(3,4)
        graph.add_edge(4,5)
        graph.add_edge(5,6)
        graph.add_edge(6,0)

        print("Growing IN")
        for x in range (7, 27):
            graph.add_edge(x, 1)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))

        print("Growing OUT")
        for x in range (27, 47):
            graph.add_edge(2, x)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))
        
        print("Growing SCC")
        for x in range (7, 27):
            graph.add_edge(1, x)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))
        
            graph.add_edge(x+20, 2)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))

        print("Shrinking SCC")
        for x in range (7, 26):
            graph.remove_edge(1, x)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))

            graph.remove_edge(x+20, 2)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))

        print("Shrinking OUT")
        for x in range (27, 47):
            graph.remove_edge(2, x)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))

        print("Shrinking IN")
        for x in range (7, 27):
            graph.remove_edge(x, 1)
            iteration += 1
            self.plot_graph(graph, name+str(iteration))
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow & Shrink\t(%.2fs)" %duration)

    def test_bowtie_plot_growAndShrinkComponentsWithObjectCopy(self):
        print("Plotting Test Start:\tGrow & Shrink (deep copy)")
        start_time = time.time()
        name = "growAndShrinkComponentsWithObjectCopy"
        graph = Graph()
        
        graphs = []
        # create graphs collection instance 
        # is needed to create plotting instance
        gc = GraphCollection(name)

        # SCC
        graph.add_edge(0,1)
        graph.add_edge(1,2)
        graph.add_edge(2,3)
        graph.add_edge(3,4)
        graph.add_edge(4,5)
        graph.add_edge(5,6)
        graph.add_edge(6,0)

        print("Growing IN")
        for x in range (7, 27):
            graph.add_edge(x, 1)
            gc.append(copy.deepcopy(graph))
        
        print("Growing OUT")
        for x in range (27, 47):
            graph.add_edge(2, x)
            gc.append(copy.deepcopy(graph))
        
        print("Growing SCC")
        for x in range (7, 27):
            graph.add_edge(1, x)
            gc.append(copy.deepcopy(graph))
        
            graph.add_edge(x+20, 2)
            gc.append(copy.deepcopy(graph))

        print("Shrinking SCC")
        for x in range (7, 26):
            graph.remove_edge(1, x)
            gc.append(copy.deepcopy(graph))

            graph.remove_edge(x+20, 2)
            gc.append(copy.deepcopy(graph))

        print("Shrinking OUT")
        for x in range (27, 47):
            graph.remove_edge(2, x)
            gc.append(copy.deepcopy(graph))

        print("Shrinking IN")
        for x in range (7, 27):
            graph.remove_edge(x, 1)
            gc.append(copy.deepcopy(graph))

        graphs.append(gc)
        print("Plotting Graphs")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tGrow & Shrink (deep copy)\t(%.2fs)" %duration)

    def test_bowtie_plot_networkx_generators(self):
        print("Plotting Test Start:\tnx Generators")
        start_time = time.time()
        name = "nxGenerators"
        graphs = []
        # create graphs collection instance 
        # is needed to create plotting instance
        gc = GraphCollection(name)
        
        graph = Graph()
        
        print("Generating 'Complete Graph'")
        complete_graph = nx.complete_graph(10)
        graph.add_edges_from(complete_graph.edges())
        gc.append(copy.deepcopy(graph))
        graph.clear()
        
        print("Generating 'Dorogovtsev Graph'")
        dorogovtsev_graph = nx.dorogovtsev_goltsev_mendes_graph(10)
        graph.add_edges_from(dorogovtsev_graph.edges())
        gc.append(copy.deepcopy(graph))
        graph.clear()

        print("Generating 'Barabasi Graph'")
        barabasi_graph = nx.barabasi_albert_graph(40, 4)
        graph.add_edges_from(barabasi_graph.edges())
        gc.append(copy.deepcopy(graph))
        graph.clear()
        
        print("Generating 'Power Law Graph'")
        power_law_graph = nx.powerlaw_cluster_graph(40, 3, 0.7)
        graph.add_edges_from(power_law_graph.edges())        
        gc.append(copy.deepcopy(graph))
        graph.clear()
        
        graphs.append(gc)
        print("Starting to plot the Graph Collection")
        self.plot_graph_collection(graphs, name)
        duration = (time.time() - start_time)
        print("Plotting Test End:\tnx Generators\t(%.2fs)" %duration)
    
    def test_bowtie_plot_10_nodes_gt(self):
        print("Plotting Test Start:\t10 Nodes (GT)")
        start_time = time.time()

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

        self.plot_graph(graph, "10_nodes_gt")
        duration = (time.time() - start_time)
        print("Plotting Test End:\t10 Nodes (GT)\t(%.2fs)" %duration)

    def test_bowtie_plot_20_nodes_gt(self):
        print("Plotting Test Start:\t20 Nodes (GT)")
        start_time = time.time()

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

        
        self.plot_graph(graph, "20_nodes_gt")
        duration = (time.time() - start_time)
        print("Plotting Test End:\t20 Nodes (GT)\t(%.2fs)" %duration)

    def test_bowtie_plot_30_nodes_gt(self):
        print("Plotting Test Start:\t30 Nodes (GT)")
        start_time = time.time()

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

        self.plot_graph(graph, "30_nodes_gt")
        duration = (time.time() - start_time)
        print("Plotting Test End:\t30 Nodes (GT)\t(%.2fs)" %duration)

    def test_bowtie_plot_40_nodes_gt(self):
        print("Plotting Test Start:\t40 Nodes (GT)")
        start_time = time.time()

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

        self.plot_graph(graph, "40_nodes_gt")
        duration = (time.time() - start_time)
        print("Plotting Test End:\t40 Nodes (GT)\t(%.2fs)" %duration)

    def test_bowtie_plot_big_in_gt(self):
        print("Plotting Test Start:\tBig In (GT)")
        start_time = time.time()

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

                
        self.plot_graph(graph, "big_in_gt")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig In (GT)\t(%.2fs)" %duration)

    def test_bowtie_plot_big_out_gt(self):
        print("Plotting Test Start:\tBig Out (GT)")
        start_time = time.time()

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

                
        self.plot_graph(graph, "big_out_gt")
        duration = (time.time() - start_time)
        print("Plotting Test End:\tBig Out (GT)\t(%.2fs)" %duration)

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
        for g in graphs:
            g.compute()
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
        p = TestPlotting('test_bowtie_plot_tiny')
        
        """uncomment the test cases you want to run"""
        #p.test_bowtie_plot_generator()
        #p.test_bowtie_plot_tiny()
        #p.test_bowtie_plot_small()
        
        #p.test_bowtie_plot_medium()
        
        #p.test_bowtie_plot_bigIN()
        #p.test_bowtie_plot_bigSCC()
        #p.test_bowtie_plot_growIN()
        #p.test_bowtie_plot_growComponents()
        #p.test_bowtie_plot_growAndShrinkComponents()
        #p.test_bowtie_plot_growAndShrinkComponentsWithObjectCopy()
        #p.test_bowtie_plot_networkx_generators()

        """graph_tool test cases here"""
        #p.test_bowtie_plot_medium_gt()
        p.test_bowtie_plot_10_nodes_gt()
        p.test_bowtie_plot_20_nodes_gt()
        p.test_bowtie_plot_30_nodes_gt()
        p.test_bowtie_plot_40_nodes_gt()

    # run all stats and plotting test cases
    #test_stats()
    test_plotting()
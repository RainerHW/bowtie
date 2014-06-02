# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import unittest
import generators

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
    def test_bowtie_plot_generator(self):
        """
        Creates Graphs using the generator class
        """
        n = 500
        m = 1
        graph = generators.barabasi_albert_graph_directed(n, m)
        self.call_plot_graph_with(Graph(graph), "barabasi_albert_graph_directed")

    def test_bowtie_plot_bigIN(self):
        """
        Creates a graph with a big IN-Component and a small SCC component
        """
        print("Plotting Test Start: Big IN")
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
        self.call_plot_graph_with(graph, "bigIN", 0)
        print("Plotting Test End: Big IN")

    def test_bowtie_plot_bigSCC(self):
        print("Plotting Test Start: Big SCC")

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
        self.call_plot_graph_with(graph, "bigSCC", 0)
        print("Plotting Test End: Big SCC")

    def test_bowtie_plot_growComponents(self):
        print("Plotting Test Start: Growing Components")
        name = "growComponents"
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
            self.call_plot_graph_with(graph, name, iteration)

        print("Growing OUT")
        for x in range (27, 47):
            graph.add_edge(2, x)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)
        
        print("Growing SCC")
        for x in range (7, 27):
            graph.add_edge(1, x)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)
        
            graph.add_edge(x+20, 2)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)

        print("Plotting Test End: Growing Components")

    def test_bowtie_plot_growAndShrinkComponents(self):
        print("Plotting Test Start: Growing & Shrinking Components")
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
            self.call_plot_graph_with(graph, name, iteration)

        print("Growing OUT")
        for x in range (27, 47):
            graph.add_edge(2, x)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)
        
        print("Growing SCC")
        for x in range (7, 27):
            graph.add_edge(1, x)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)
        
            graph.add_edge(x+20, 2)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)

        print("Shrinking SCC")
        for x in range (7, 26):
            graph.remove_edge(1, x)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)

            graph.remove_edge(x+20, 2)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)

        print("Shrinking OUT")
        for x in range (27, 47):
            graph.remove_edge(2, x)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)

        print("Shrinking IN")
        for x in range (7, 27):
            graph.remove_edge(x, 1)
            iteration += 1
            self.call_plot_graph_with(graph, name, iteration)

        print("Plotting Test End: Growing & Shrinking Components")
    
    def test_bowtie_plot_tiny(self):
        print("Plotting Test Start: Tiny Graph")
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

        self.call_plot_graph_with(graph, "tiny", 0)
        print("Plotting Test End: Tiny Graph")

    def test_bowtie_plot_small(self):
        print("Plotting Test Start: Small Graph")
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

        self.call_plot_graph_with(graph, "small", 0)
        print("Plotting Test End: Small Graph")

    def test_bowtie_plot_medium(self):
        print("Plotting Test Start: Medium Graph")
        graph = Graph()

        # build scc
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
        # build in
        # build out
        self.call_plot_graph_with(graph, "medium", 0)
        print("Plotting Test End: Medium Graph")
    
    def call_plot_graph_with(self, graph, testcase, iteration):        
        """
        Helper Method: Create the GC, Plotting object and call bowtieplot
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
        
        #compute statistics for graphs
        for g in graphs:
            g.compute()
        
        # create Plotting instance and call the plotting method
        plotting = Plotting(graphs)
        plotting.bowtieplot(testcase, iteration)

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
        #p.test_bowtie_plot_bigIN()
        #p.test_bowtie_plot_bigSCC()
        #p.test_bowtie_plot_growComponents()
        p.test_bowtie_plot_growAndShrinkComponents()
        #p.test_bowtie_plot_tiny()
        #p.test_bowtie_plot_small()
        #p.test_bowtie_plot_medium()

    # run all stats and plotting test cases
    test_stats()
    test_plotting()
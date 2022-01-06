import numpy as np
import heapq
import time


class DistancesFinderHeapq:
    def __init__(self, graph, smart=True):
        self.edges = []
        for edge in graph.edges():
            self.edges.append((graph.vertex_index[edge.source()], graph.vertex_index[edge.target()]))
            
        self.n_vertex = graph.num_vertices() 
        self.adj_matrix = []
        for i in range(self.n_vertex):
            # list of {vertex_id: distance, ..}
            self.adj_matrix.append(dict((x, -1) for x in graph.get_out_neighbors(i)))

        self.adj__ = np.full((self.n_vertex, self.n_vertex), float('inf'))
        for i in range(self.n_vertex):
            self.adj__[i,i] = 0 
#         for i in range(self.n_vertex):
#             for j in graph.get_out_neighbors(i):
#                 self.adj___[i, j] = 
        self.outs__ = []
        for i in range(self.n_vertex):
            self.outs__.append(np.array([x for x in graph.get_out_neighbors(i)], dtype=int))

        self.tree_order = [None for _ in range(self.n_vertex)]
        self.pred_map = [None for _ in range(self.n_vertex)]
        
        self.smart = smart
        print(f'set smart={smart} !')
        
        self.first_loop = [[] for _ in range(self.n_vertex)]
        self.second_loop = [[] for _ in range(self.n_vertex)]
        self.heapify_time = [[] for _ in range(self.n_vertex)]
        self.main_loop = [[] for _ in range(self.n_vertex)]
        self.last_loop = [[] for _ in range(self.n_vertex)]
        self.n = 0
        
        self.first_time_time = [[] for _ in range(self.n_vertex)]
        
        self.n_unchanged = [[] for _ in range(self.n_vertex)]
        self.changed = [[] for _ in range(self.n_vertex)]
        
    
    def first_time(self, source, target):
        #self.n += 1
        tic = time.time()
        shortest_distances = np.full(self.n_vertex, float('inf'))
        shortest_distances[source] = 0

        tree_order = []
        pred_map = np.arange(self.n_vertex)
        
        visited = np.zeros(self.n_vertex, dtype=bool)
        # distance, (vertex, pred_vertex)
        min_heap = [(0, (source, source)),]
        while min_heap:
            distance, (vertex, pred_vertex) = heapq.heappop(min_heap)
            
            if visited[vertex]:
                continue
            
            visited[vertex] = True
            tree_order.append((vertex, pred_vertex))
            pred_map[vertex] = pred_vertex

            for neighbour, nb_distance in self.adj_matrix[vertex].items():
                d = distance + nb_distance
                if d < shortest_distances[neighbour]:
                    shortest_distances[neighbour] = d
                    heapq.heappush(min_heap, (d, (neighbour, vertex)))
                # mb Better to update object .... in BinTree ??
        #self.first_time_time += (time.time() - tic - self.first_time_time) / (self.n + 1)
        self.first_time_time[source].append(time.time() - tic)
        #print(f'FT: {self.first_time_time*10**5:.2f}')
        
        self.tree_order[source] = tree_order
        self.pred_map[source] = pred_map
     
        distances = []
        for t in target:
            distances.append(shortest_distances[t])
        
        return np.array(distances), self.pred_map[source]        


    def recompute_distances(self, source, target):
        #print('First time in recompute')
        #raise KeybordInterupt
        pred_map = self.pred_map[source]
        shortest_distances = np.full(self.n_vertex, float('inf'))
        shortest_distances[source] = 0

        # STEP 1
        self.n += 1
        tic = time.time()
        # recompute distances according to last tree order
        for vertex, pred_vertex in self.tree_order[source]:
            shortest_distances[vertex] = shortest_distances[pred_vertex] + self.adj__[pred_vertex, vertex]
        self.first_loop[source].append(time.time() - tic)

        # STEP 2
        tic = time.time()
        # find some vertexes with changed shortest paths if any exists
        min_heap = []
        for vertex, pred_vertex in self.tree_order[source]:
            for neighbour, nb_distance in self.adj_lists[vertex]:
                d = shortest_distances[vertex] + nb_distance
                if d < shortest_distances[neighbour]:
                    shortest_distances[neighbour] = d
                    heapq.heappush(min_heap, (d, (neighbour, vertex)))
        self.second_loop[source].append(time.time() - tic)
        
        # STEP 3
        tic = time.time()
        visited = np.zeros(self.n_vertex, dtype=bool)
        tree_order = []
        # recompute shotest distacnes for all changed paths
        while min_heap:
            distance, (vertex, pred_vertex) = heapq.heappop(min_heap)
            
            if visited[vertex]:
                continue
            
            visited[vertex] = True
            tree_order.append((vertex, pred_vertex))
            pred_map[vertex] = pred_vertex          
            
            for neighbour, nb_distance in self.adj_matrix[vertex].items():
                d = distance + nb_distance
                if d < shortest_distances[neighbour]:
                    shortest_distances[neighbour] = d
                    heapq.heappush(min_heap, (d, (neighbour, vertex)))
        self.main_loop[source].append(time.time() - tic)

        # STEP 4
        tic = time.time()
        # update tree order
        unchanged_tree_order = []
        for vertex, pred_vertex in self.tree_order[source]:
            if not visited[vertex]:
                unchanged_tree_order.append((vertex, pred_vertex))
        self.last_loop[source].append(time.time() - tic)
        
        self.n_unchanged[source].append(len(unchanged_tree_order))
        self.changed[source].append(len(tree_order))
    
        self.tree_order[source] = unchanged_tree_order + tree_order
        self.pred_map[source] = pred_map
        
        distances = []
        for t in target:
            distances.append(shortest_distances[t])
        
        #print('did RECOMPUTE 1 time for ', source, ' source')
#         if source == 0:
#             print(self.tree_order[source], shortest_distances)
#         print(f'H: {self.heapify_time*10**5:.2f}', end='   ')
#         print(f'{self.first_loop*10**5:.2f}, {self.second_loop*10**5:.2f}, {self.main_loop*10**5:.2f}, '
#               f'{self.last_loop*10**5:.2f}')
        return np.array(distances), self.pred_map[source]
        
    def update_weights(self, weights):
        assert len(list(weights)) == len(self.edges)
        for w, edge in zip(weights, self.edges):
            s, t = edge
            self.adj_matrix[s][t] = w
            
            self.adj__[s, t] = w
            
        self.adj_lists = []
        for i, dct in enumerate(self.adj_matrix):
            self.adj_lists.append(list(dct.items()))

    def shortest_distance(self, g, source, target, weights, pred_map = True):
        self.update_weights(weights)
        
        if (self.tree_order[source] is None) or not self.smart:
            return self.first_time(source, target)
        
        return self.recompute_distances(source, target)

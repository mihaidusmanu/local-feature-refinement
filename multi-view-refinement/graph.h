// Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu (mihai.dusmanu@inf.ethz.ch)

#include <cmath>

#include <vector>

#include <string>

#include <Eigen/Core>

class Edge {
    public:
        Edge(size_t, size_t, double, std::vector<double>&);

        size_t src_idx = -1;
        size_t dst_idx = -1;
        double sim;
        std::vector<double> flow;
};

class PatchNode {
    public:
        PatchNode(std::string, int);
        void add_edge(PatchNode*, double, std::vector<double>&);

        std::string image_name;
        size_t feature_idx = -1;
        size_t node_idx = -1;
        std::vector<Edge*> out_edges;
        std::vector<Edge*> in_edges;
};

class Graph {
    public:
        size_t add_node(PatchNode*);
        std::vector<size_t> get_input_degrees();
        std::vector<size_t> get_output_degrees();
        std::vector<std::pair<double, size_t>> get_scores();
        
        std::vector<PatchNode*> nodes;
};

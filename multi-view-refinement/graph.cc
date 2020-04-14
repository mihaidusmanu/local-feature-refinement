// Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu (mihai.dusmanu@inf.ethz.ch)

#include "graph.h"

Edge::Edge(size_t source_idx, size_t destination_idx, double similarity, std::vector<double>& flow_array) {
    src_idx = source_idx;
    dst_idx = destination_idx;
    sim = similarity; 
    flow = flow_array;
}

PatchNode::PatchNode(std::string im_name, int feat_idx) {
    image_name = im_name;
    feature_idx = feat_idx;
}

void PatchNode::add_edge(PatchNode* neighbor, double similarity, std::vector<double>& flow_array) {
    Edge* edge = new Edge(this->node_idx, neighbor->node_idx, similarity, flow_array);

    out_edges.push_back(edge);
    
    neighbor->in_edges.push_back(edge);
}

size_t Graph::add_node(PatchNode* node) {
    nodes.push_back(node);
    node->node_idx = nodes.size() - 1;
    return node->node_idx;
}

std::vector<size_t> Graph::get_input_degrees() {
    std::vector<size_t> input_degrees;
    for (auto &node : nodes) {
        input_degrees.push_back(node->in_edges.size());
    }
    return input_degrees;
}

std::vector<size_t> Graph::get_output_degrees() {
    std::vector<size_t> output_degrees;
    for (auto &node : nodes) {
        output_degrees.push_back(node->out_edges.size());
    }
    return output_degrees;
}

std::vector<std::pair<double, size_t>> Graph::get_scores() {
    std::vector<std::pair<double, size_t>> scores;
    for (auto &node : nodes) {
        double s = 0.;
        for (auto &edge : node->out_edges) {
            s += edge->sim;
        }
        scores.push_back(std::make_pair(s, node->node_idx));
    }
    return scores;
}
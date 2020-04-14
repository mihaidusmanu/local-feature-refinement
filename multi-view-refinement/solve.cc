// Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu (mihai.dusmanu@inf.ethz.ch)

#include <cmath>

#include <vector>

#include <map>

#include <iostream>

#include <unistd.h>
#include <fcntl.h>

#include <fstream>

#include <chrono>

#include <limits>

#include "boost/program_options.hpp"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/coded_stream.h"

#include "colmap/util/threading.h"
#include "colmap/base/graph_cut.h"

#include "graph.h"
#include "cost.cc"
#include "types.pb.h"

namespace po = boost::program_options;

using namespace google::protobuf::io;
using namespace colmap;

typedef std::map<std::pair<std::string, size_t>, size_t> graph_map;
typedef std::set<std::pair<std::string, std::string>> image_pair_set;
typedef std::map<std::string, size_t> image_map;
typedef std::tuple<double, size_t, size_t> edge;

typedef BiquadraticInterpolator Interpolator;

const size_t kNumSamples = 3;

bool file_exists(std::string file_name)
{
    std::ifstream infile(file_name);
    return infile.good();
}

PatchNode* find_or_create_node(std::string image_name, size_t feature_idx, graph_map* nodes, Graph* graph) {
    graph_map::iterator it = nodes->find(std::make_pair(image_name, feature_idx));

    if (it != nodes->end()) {
        return graph->nodes[it->second];
    } else {
        PatchNode* node = new PatchNode(image_name, feature_idx);
        size_t node_idx = graph->add_node(node);
        node->node_idx = node_idx;
        nodes->insert(std::make_pair(std::make_pair(image_name, feature_idx), node_idx));
        return node;
    }
}

size_t union_find_get_root(
        const size_t node_idx,
        std::vector<int>& parent_nodes
) {
    if (parent_nodes[node_idx] == -1) {
        return node_idx;
    }
    // Union-find path compression heuristic.
    parent_nodes[node_idx] = union_find_get_root(parent_nodes[node_idx], parent_nodes);
    return parent_nodes[node_idx];
}

void create_and_solve_problem(
        const Graph& graph,
        const std::vector<size_t>& track_idx_container,
        std::vector<Eigen::Vector2d>& positions,
        const std::vector<bool>& is_root,
        const std::vector<size_t>& component_idx_container,
        const std::vector<size_t>& nodes_in_component,
        const size_t solver_num_threads
) {
    // Extra parameters.
    const double bound = 1.;

    // Create the problem.
    ceres::Problem problem;
    
    size_t n_intra_edges = 0;
    size_t n_inter_edges = 0;
    std::set<size_t> tracks;
    std::vector<bool> will_be_optimized(nodes_in_component.size(), false);
    for (size_t node_idx_in_component = 0; node_idx_in_component < nodes_in_component.size(); ++node_idx_in_component) {
        size_t node_idx = nodes_in_component[node_idx_in_component];
        tracks.insert(track_idx_container[node_idx]);
        PatchNode* node = graph.nodes[node_idx];
        for (auto &edge : node->out_edges) {
            Interpolator interpolator(edge->flow, 2);

            if (track_idx_container[node_idx] == track_idx_container[edge->dst_idx]) {
                ++n_intra_edges;
                problem.AddResidualBlock(
                    InterpolatedCostFunctor<Interpolator>::Create(
                        std::move(interpolator)
                    ),
                    new ceres::ScaledLoss(new ceres::CauchyLoss(0.25), edge->sim, ceres::TAKE_OWNERSHIP), 
                    positions[node_idx].data(), positions[edge->dst_idx].data()
                );
            } else if (component_idx_container[node_idx] == component_idx_container[edge->dst_idx]) {
                ++n_inter_edges;
                problem.AddResidualBlock(
                    InterpolatedCostFunctor<Interpolator>::Create(
                        std::move(interpolator)
                    ),
                    new ceres::ScaledLoss(new ceres::TukeyLoss(0.0625), edge->sim, ceres::TAKE_OWNERSHIP),
                    positions[node_idx].data(), positions[edge->dst_idx].data()
                );
            } else {
                continue;
            }

            will_be_optimized[node_idx_in_component] = true;
        }
    }

    for (size_t node_idx_in_component = 0; node_idx_in_component < nodes_in_component.size(); ++node_idx_in_component) {
        size_t node_idx = nodes_in_component[node_idx_in_component];
        if (will_be_optimized[node_idx_in_component]) {
            if (is_root[node_idx]) {
                problem.SetParameterBlockConstant(positions[node_idx].data());
            } else {
                problem.SetParameterLowerBound(positions[node_idx].data(), 0, -bound);
                problem.SetParameterUpperBound(positions[node_idx].data(), 0, bound);
                problem.SetParameterLowerBound(positions[node_idx].data(), 1, -bound);
                problem.SetParameterUpperBound(positions[node_idx].data(), 1, bound);
            }
        }
    }

    // CERES options.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = solver_num_threads;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    options.max_num_consecutive_invalid_steps = 10;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-4;

    // Solve the problem.
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
}

void bfs(
        const size_t node_idx,
        const size_t current_component_idx,
        const std::vector<std::unordered_map<size_t, double>>& edges,
        std::vector<size_t>& component_idx_container
) {
    std::queue<size_t> nodes_queue;
    nodes_queue.push(node_idx);
    component_idx_container[node_idx] = current_component_idx;

    while (!nodes_queue.empty()) {
        const size_t current_node_idx = nodes_queue.front();
        nodes_queue.pop();
        for (auto& it : edges[current_node_idx]) {
            if (component_idx_container[it.first] <= current_component_idx) {
                continue;
            }
            nodes_queue.push(it.first);
            component_idx_container[it.first] = current_component_idx;
        }
    }
}

std::unordered_map<int, int> recursive_graph_cut(
        const std::vector<std::pair<int, int>>& edges,
        const std::vector<int>& weights,
        const std::vector<size_t>& node_weights,
        const size_t max_subset_weight
) {
    const size_t n_subsets = 2;
    const std::unordered_map<int, int> nodes_to_subsets_map = colmap::ComputeNormalizedMinGraphCut(edges, weights, n_subsets);
    
    std::vector<size_t> subset_weights(n_subsets, 0);
    std::vector<std::vector<int>> nodes_in_subset(n_subsets);

    for (auto& it : nodes_to_subsets_map) {
        subset_weights[it.second] += node_weights[it.first];
        nodes_in_subset[it.second].push_back(it.first);
    }

    int max_subset_idx = 0;
    std::unordered_map<int, int> final_nodes_to_subsets_map;
    for (size_t subset_idx = 0; subset_idx < n_subsets; ++subset_idx) {
        if (subset_weights[subset_idx] <= max_subset_weight) {
            for (auto& node_idx : nodes_in_subset[subset_idx]) {
                final_nodes_to_subsets_map.insert(std::make_pair(node_idx, max_subset_idx));
            }
            ++max_subset_idx;
            continue;
        }

        std::vector<std::pair<int, int>> edges_in_current_subset;
        std::vector<int> weights_in_current_subset;
        for(size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
            const int node1 = edges[edge_idx].first;
            const int node2 = edges[edge_idx].second;
            const auto subset_idx1 = nodes_to_subsets_map.find(node1);
            assert(subset_idx1 != nodes_to_subsets_map.end());
            const auto subset_idx2 = nodes_to_subsets_map.find(node2);
            assert(subset_idx2 != nodes_to_subsets_map.end());

            if (subset_idx1->second == subset_idx && subset_idx2->second == subset_idx) {
                edges_in_current_subset.push_back(edges[edge_idx]);
                weights_in_current_subset.push_back(weights[edge_idx]);
            }
        }

        if (edges_in_current_subset.size() > 0) {
            std::unordered_map<int, int> new_nodes_to_subsets_map = recursive_graph_cut(edges_in_current_subset, weights_in_current_subset, node_weights, max_subset_weight);

            int new_max_subset_idx = max_subset_idx;
            for (auto& it : new_nodes_to_subsets_map) {
                final_nodes_to_subsets_map.insert(std::make_pair(it.first, max_subset_idx + it.second));
                new_max_subset_idx = std::max(new_max_subset_idx, max_subset_idx + it.second);
            }
            max_subset_idx = new_max_subset_idx + 1;
        }

        for (auto& it : nodes_in_subset[subset_idx]) {
            if (final_nodes_to_subsets_map.find(it) != final_nodes_to_subsets_map.end()) {
                continue;
            }
            final_nodes_to_subsets_map.insert(std::make_pair(it, max_subset_idx));
            ++max_subset_idx;
        }
    }

    return final_nodes_to_subsets_map;
}

std::vector<size_t> separate_meta_graph(
        const Graph& graph,
        const std::vector<size_t>& track_idx_container,
        const size_t max_nodes_in_component
) {
    // Recover # nodes and # tracks.
    const size_t n_nodes = graph.nodes.size();
    const size_t n_tracks = (*std::max_element(track_idx_container.begin(), track_idx_container.end())) + 1;

    // Compute the number of nodes in each track.
    std::vector<size_t> n_nodes_in_track(n_tracks, 0);
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        ++n_nodes_in_track[track_idx_container[node_idx]];
    }

    // Compute the meta-graph.
    std::vector<std::unordered_map<size_t, double>> meta_edges(n_tracks);
    std::vector<std::unordered_map<size_t, size_t>> meta_edges_extra(n_tracks);
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        PatchNode* node = graph.nodes[node_idx];
        const size_t source_track = track_idx_container[node_idx];

        for (auto &edge : node->out_edges) {
            const size_t target_track = track_idx_container[edge->dst_idx];

            if (source_track != target_track) {
                std::unordered_map<size_t, double>::iterator it = meta_edges[source_track].find(target_track);
                std::unordered_map<size_t, size_t>::iterator it_ = meta_edges_extra[source_track].find(target_track);
                if (it != meta_edges[source_track].end()) {
                    it->second += edge->sim;
                    it_->second += 1;
                } else {
                    meta_edges[source_track].insert(std::make_pair(target_track, edge->sim));
                    meta_edges_extra[source_track].insert(std::make_pair(target_track, 1));
                }
            }
        }
    }

    // Compute connected components.
    size_t n_components = 0;
    std::vector<size_t> component_idx_container(n_tracks, -1);
    for (size_t meta_node_idx = 0; meta_node_idx < n_tracks; ++meta_node_idx) {
        if (component_idx_container[meta_node_idx] < n_components) {
            continue;
        }
        bfs(meta_node_idx, n_components, meta_edges, component_idx_container);
        ++n_components;
    }

    // Compute the sizes of each connected component.
    std::vector<size_t> n_nodes_in_connected_component(n_components, 0);
    std::vector<std::vector<size_t>> meta_nodes_in_connected_component(n_components);
    for (size_t meta_node_idx = 0; meta_node_idx < n_tracks; ++meta_node_idx) {
        n_nodes_in_connected_component[component_idx_container[meta_node_idx]] += n_nodes_in_track[meta_node_idx];
        meta_nodes_in_connected_component[component_idx_container[meta_node_idx]].push_back(meta_node_idx);
    }

    // Recursive normalized graph cut.
    size_t n_gc_components = 0;
    std::vector<size_t> gc_component_idx_container(n_tracks);
    for (size_t c_idx = 0; c_idx < n_components; ++c_idx) {
        if (n_nodes_in_connected_component[c_idx] <= max_nodes_in_component) {
            for (auto& meta_node_idx : meta_nodes_in_connected_component[c_idx]) {
                gc_component_idx_container[meta_node_idx] = n_gc_components;
            }
            ++n_gc_components;
            continue;
        }

        std::vector<std::pair<int, int>> edges;
        std::vector<int> weights;
        for (auto& meta_node_idx : meta_nodes_in_connected_component[c_idx]) {
            for (auto& it : meta_edges[meta_node_idx]) {
                 // Undirected graph.
                if (meta_node_idx < it.first) {
                    edges.push_back(std::make_pair(meta_node_idx, it.first));
                    weights.push_back(static_cast<int>(100 * it.second));
                }
            }
        }

        std::unordered_map<int, int> split = recursive_graph_cut(edges, weights, n_nodes_in_track, max_nodes_in_component);
        assert(split.size() == meta_nodes_in_connected_component[c_idx].size());

        size_t new_n_gc_components = 0;
        for (auto& it : split) {
            gc_component_idx_container[it.first] = n_gc_components + static_cast<size_t>(it.second);
            new_n_gc_components = std::max(new_n_gc_components, gc_component_idx_container[it.first]);
        }
        n_gc_components = new_n_gc_components + 1;
    }

    // Compute the remaining edges after graph cut.
    std::vector<std::unordered_map<size_t, double>> post_gc_meta_edges(n_tracks);
    for (size_t meta_node_idx = 0; meta_node_idx < n_tracks; ++meta_node_idx) {
        for (auto& it : meta_edges[meta_node_idx]) {
            if (gc_component_idx_container[meta_node_idx] == gc_component_idx_container[it.first]) {
                post_gc_meta_edges[meta_node_idx].insert(it);
            }
        }
    }

    // Re-split in connected components.
    size_t n_final_components = 0;
    std::vector<size_t> final_component_idx_container(n_tracks, -1);
    for (size_t meta_node_idx = 0; meta_node_idx < n_tracks; ++meta_node_idx) {
        if (final_component_idx_container[meta_node_idx] < n_final_components) {
            continue;
        }
        bfs(meta_node_idx, n_final_components, post_gc_meta_edges, final_component_idx_container);
        ++n_final_components;
    }

    // Compute the final components for each node.
    std::vector<size_t> answer(n_nodes, -1);
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        answer[node_idx] = final_component_idx_container[track_idx_container[node_idx]];
    }

    return answer;
}

int main(int argc, char** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // Launch options.
    po::options_description desc("Options");
    desc.add_options()
        ("help", "print the help")
        ("matches_file", po::value<std::string>()->required(), "path to the matches file")
        ("output_file", po::value<std::string>()->required(), "path to the output file")
        ("n_threads", po::value<size_t>()->default_value(size_t(8), "8"), "# threads")
        ("banned_images", po::value<std::vector<std::string>>()->default_value(std::vector<std::string>(), "{}"), "banned images");
    
    po::variables_map args; 
    try { 
      po::store(po::parse_command_line(argc, argv, desc), args);
 
      if (args.count("help")) { 
        std::cout << "Patch Match graph problem solver" << std::endl << std::endl << desc; 
        return 0; 
      } 
 
      po::notify(args);
    } catch(po::error& e) { 
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cerr << desc; 
      return 1; 
    }

    std::set<std::string> banned_images(args["banned_images"].as<std::vector<std::string>>().begin(), args["banned_images"].as<std::vector<std::string>>().end());

    // Define the graph.
    graph_map nodes;
    std::vector<edge> edges;
    std::set<std::string> images_set;
    Graph graph;
    std::map<std::string, float> images_facts;

    // Read the matches.
    std::string matches_file = args["matches_file"].as<std::string>();
    std::vector<std::string> matches_files;

    if (!file_exists(matches_file)) {
        size_t part_idx = 0;
        while (file_exists(matches_file + ".part." + std::to_string(part_idx))) {
            matches_files.push_back(matches_file + ".part." + std::to_string(part_idx));
            part_idx += 1;
        }
    } else {
        matches_files.push_back(matches_file);
    }

    for (auto file : matches_files) {
        int fd = open(file.c_str(), O_RDONLY);
        ZeroCopyInputStream* raw_input = new FileInputStream(fd);
        CodedInputStream* coded_input = new CodedInputStream(raw_input);
        coded_input->SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());

        MatchingFile matching_file;
        if (!matching_file.ParseFromCodedStream(coded_input)) {
            std::cerr << "Failed to parse proto object." << std::endl;
            return -1;
        }

        for (size_t image_pair_idx = 0; image_pair_idx < matching_file.image_pairs_size(); ++image_pair_idx) {
            const MatchingFile_ImagePair& image_pair = matching_file.image_pairs(image_pair_idx);

            std::string image_name1 = image_pair.image_name1();
            std::string image_name2 = image_pair.image_name2();

            if (banned_images.find(image_name1) != banned_images.end() || banned_images.find(image_name2) != banned_images.end()) {
                continue;
            }

            images_set.insert(image_name1);
            images_facts.insert(std::make_pair(image_name1, image_pair.fact1()));
            images_set.insert(image_name2);
            images_facts.insert(std::make_pair(image_name2, image_pair.fact2()));

            for (size_t match_idx = 0; match_idx < image_pair.matches_size(); ++match_idx) {
                const MatchingFile_ImagePair_Match match = image_pair.matches(match_idx);

                size_t feature_idx1 = match.feature_idx1();
                size_t feature_idx2 = match.feature_idx2();
                double similarity = match.similarity();

                std::vector<double> flow_array1(2 * kNumSamples * kNumSamples);
                for (size_t grid_idx = 0; grid_idx < match.disp1_size(); ++grid_idx) {
                    const MatchingFile_ImagePair_Match_Displacement disp = match.disp1(grid_idx);
                    flow_array1[grid_idx * 2] = disp.di();
                    flow_array1[grid_idx * 2 + 1] = disp.dj();
                }

                std::vector<double> flow_array2(2 * kNumSamples * kNumSamples);
                for (size_t grid_idx = 0; grid_idx < match.disp2_size(); ++grid_idx) {
                    const MatchingFile_ImagePair_Match_Displacement disp = match.disp2(grid_idx);
                    flow_array2[grid_idx * 2] = disp.di();
                    flow_array2[grid_idx * 2 + 1] = disp.dj();
                }

                PatchNode* node1 = find_or_create_node(image_name1, feature_idx1, &nodes, &graph);
                PatchNode* node2 = find_or_create_node(image_name2, feature_idx2, &nodes, &graph);
                edges.push_back(std::make_tuple(similarity, node1->node_idx, node2->node_idx));
                node1->add_edge(node2, similarity, flow_array2);
                node2->add_edge(node1, similarity, flow_array1);
            }
        }
    }

    size_t n_nodes = graph.nodes.size();
    std::cout << "# graph nodes:" << " " << n_nodes << std::endl;
    std::cout << "# graph edges:" << " " << edges.size() * 2 << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    // Build the MSF.
    std::sort(edges.begin(), edges.end());
    std::reverse(edges.begin(), edges.end());

    std::vector<int> parent_nodes(n_nodes, -1);
    std::vector<std::set<std::string>> images_in_track(n_nodes);

    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        images_in_track[node_idx].insert(graph.nodes[node_idx]->image_name);
    }
    
    for (auto it : edges) {
        size_t node_idx1 = std::get<1>(it);
        size_t node_idx2 = std::get<2>(it);

        size_t root1 = union_find_get_root(node_idx1, parent_nodes);
        size_t root2 = union_find_get_root(node_idx2, parent_nodes);

        if (root1 != root2) {
            std::set<std::string> intersection;
            std::set_intersection(images_in_track[root1].begin(), images_in_track[root1].end(), images_in_track[root2].begin(), images_in_track[root2].end(), std::inserter(intersection, intersection.begin()));
            if (intersection.size() != 0) {
                continue;
            }
            // Union-find merging heuristic.
            if (images_in_track[root1].size() < images_in_track[root2].size()) {
                parent_nodes[root1] = root2;
                images_in_track[root2].insert(images_in_track[root1].begin(), images_in_track[root1].end());
                images_in_track[root1].clear();
            } else {
                parent_nodes[root2] = root1;
                images_in_track[root1].insert(images_in_track[root2].begin(), images_in_track[root2].end());
                images_in_track[root2].clear();
            }
        }
    }

    // Compute the tracks.
    std::vector<size_t> track_idx_container(n_nodes, -1);

    size_t n_tracks = 0;
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        if (parent_nodes[node_idx] == -1) {
            track_idx_container[node_idx] = n_tracks++;
        }
    }
    std::cout << "# tracks:" << " " << n_tracks << std::endl;
    
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        if (track_idx_container[node_idx] != -1) {
            continue;
        }
        track_idx_container[node_idx] = track_idx_container[union_find_get_root(node_idx, parent_nodes)];
    }

    std::vector<size_t> elements_per_track(n_tracks, 0);
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        ++elements_per_track[track_idx_container[node_idx]];
    }

    size_t max_track_size = *(std::max_element(elements_per_track.begin(), elements_per_track.end()));
    std::cout << "max track size:" << " " << max_track_size << std::endl;

    // Find the root nodes.
    std::vector<std::pair<double, size_t>> scores;

    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        PatchNode* node = graph.nodes[node_idx];

        double score = 0.;
        for (auto &edge : node->out_edges) {
            if (track_idx_container[node_idx] == track_idx_container[edge->dst_idx]) {
                score += edge->sim;
            }
        }

        scores.push_back(std::make_pair(score, node_idx));
    }

    std::sort(scores.begin(), scores.end());
    std::reverse(scores.begin(), scores.end());

    std::vector<bool> is_root(n_nodes, false);
    std::vector<bool> has_root(n_tracks, false);

    for (auto it : scores) {
        size_t node_idx = it.second;

        if (has_root[track_idx_container[node_idx]]) {
            continue;
        }
        
        is_root[node_idx] = true;
        has_root[track_idx_container[node_idx]] = true;
    }

    // Split graph into components.
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<size_t> component_idx_container = separate_meta_graph(graph, track_idx_container, images_set.size());
    size_t n_components = (*std::max_element(component_idx_container.begin(), component_idx_container.end())) + 1;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Graph-cut time:" << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    std::cout << "# components:" << " " << n_components << std::endl;

    // Separate nodes by tracks.
    std::vector<std::vector<size_t>> nodes_in_component(n_components);
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        nodes_in_component[component_idx_container[node_idx]].push_back(node_idx);
    }

    std::vector<std::pair<size_t, size_t>> component_sizes;
    for (size_t problem_idx = 0; problem_idx < n_components; ++problem_idx) {
        component_sizes.push_back(std::make_pair(nodes_in_component[problem_idx].size(), problem_idx));
    }
    std::sort(component_sizes.begin(), component_sizes.end());
    std::reverse(component_sizes.begin(), component_sizes.end());

    std::cout << "max component size:" << " " << component_sizes[0].first << std::endl;

    // Problem initialization.
    std::vector<Eigen::Vector2d> positions;
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        positions.push_back(Eigen::Vector2d::Zero());
    }

    // Solve each track independently.
    t1 = std::chrono::high_resolution_clock::now();
    
    colmap::ThreadPool thread_pool(args["n_threads"].as<size_t>());
    for (auto& it : component_sizes) {
        if (it.first == 1) {
            // Skip components with only one node.
            continue;
        }
        const size_t problem_idx = it.second;
        thread_pool.AddTask(
            create_and_solve_problem,
            std::ref(graph),
            std::ref(track_idx_container),
            std::ref(positions),
            std::ref(is_root),
            std::ref(component_idx_container),
            std::ref(nodes_in_component[problem_idx]),
            1
        );
    }
    thread_pool.Wait();

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Solver time:" << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time:" << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Build the proto object.
    image_map images;
    size_t nb_outside = 0;
    SolutionFile solution_file = SolutionFile();
    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        PatchNode* node = graph.nodes[node_idx];
        image_map::iterator it = images.find(node->image_name);

        SolutionFile_Image* image = NULL;
        if (it != images.end()) {
            image = solution_file.mutable_images(it->second);
        } else {
            image = solution_file.add_images();
            images.insert(std::make_pair(node->image_name, solution_file.images_size() - 1));
            image->set_image_name(node->image_name);
            image->set_fact(images_facts[node->image_name]);
        }

        SolutionFile_Image_Displacement* disp = image->add_displacements();
        disp->set_feature_idx(node->feature_idx);
        disp->set_di(positions[node_idx][0]);
        disp->set_dj(positions[node_idx][1]);

        if (std::abs(positions[node_idx][1]) > 0.5 || std::abs(positions[node_idx][0]) > 0.5) {
            ++nb_outside;
        }
    }
    std::cout << "# points with at least one coordinate > 0.5:" << " " <<  nb_outside << std::endl;

    // Save the results to disk.
    std::ofstream output_file(args["output_file"].as<std::string>(), std::ios::trunc | std::ios::binary);
    if (!solution_file.SerializeToOstream(&output_file)) {
        std::cerr << "Failed to write proto object." << std::endl;
        return -1;
    }

    output_file.close();

    return 0;
}

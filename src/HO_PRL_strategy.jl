module HO_PRL_strategy

using Revise
using StatsBase
using InvertedIndices
using Distributions
using Random
using LightGraphs
using StaticArrays
using Random
using TimerOutputs
using DelimitedFiles




### initialize players' strategies ################################
function init_plrs_strats_fixnumber(num_plrs::Int64, rhoi::Float32)
    num_coops = Int(round(num_plrs*rhoi))
    strats = zeros(Int64, num_plrs)

    for coop in 1:num_coops
        coop_ID = rand(1:length(strats))
        while strats[coop_ID] > 0
            coop_ID = rand(1:length(strats))
        end
        strats[coop_ID] = 1
    end
    return strats
end
##################################################################


### create ER hypergraph #########################################
function ER_hypergraph(num_plrs ::Int64, ktot::Int64, delta::Float32)

    edge_list = Array{Tuple{Int64, Int64},1}()

    ###### higher_vec[i] contains all the simpleces insinsting on node i, as a 2-ple (the index of the other two nodes)
    ###### pairw_vec[i] contains a list of all the neighbors interacting in pairwise interaction
    higher_vec = [Vector{Tuple{Int64, Int64}}() for node in 1:num_plrs]
    pairw_vec = [Vector{Int64}() for node in 1:num_plrs]

    
    k_pairwise = ktot*(1-delta)   ###average number of pairwise interaction
    k_simplex = ktot*(delta)      ###average number of 2-simplex interaction
    prob_edge = k_pairwise / (num_plrs-1)
    prob_simpl = k_simplex*2/((num_plrs-1)*(num_plrs-2))


    count_pairw = 0
    for i in 1:num_plrs, j in i+1:num_plrs
        if rand() < prob_edge
            push!(edge_list, (i,j))
            push!(pairw_vec[i], j)
            push!(pairw_vec[j], i)
            count_pairw += 2
        end
        for k in j+1:num_plrs
            if rand() < prob_simpl
                push!(higher_vec[i],(j,k))
                push!(higher_vec[j],(i,k))
                push!(higher_vec[k],(i,j))
                push!(edge_list, (i,j))
                push!(edge_list, (j,k))
                push!(edge_list, (i,k))
            end
        end
    end
    el = Edge.(edge_list)
    graph = SimpleGraph(el)
    if !is_connected(graph)
        println("The graph is not connected, I will take the largest component")
        components_ls = connected_components(graph)
        largest_comp_size = maximum(length.(components_ls))
        largest_comps = [component for component in components_ls if length(component) == largest_comp_size]
        if length(largest_comps) > 1
            println("There are more components of maximum size; I will take by default the first one")
        end
        subgraph_nodes = largest_comps[1]
        
        graph2, vmap = induced_subgraph(graph, subgraph_nodes)
   
        vmap_inverse = zeros(Int64, nv(graph))
        for node in vertices(graph2)
            vmap_inverse[vmap[node]] = node
        end
        count_simpl = 0
        new_higher_vec = [Vector{Tuple{Int64, Int64}}() for node in 1:nv(graph2)]
        new_pairw_vec = [Vector{Int64}() for node in 1:nv(graph2)]
        for (nodeID, simpls_4node) in enumerate(higher_vec)
            for simplx in simpls_4node
                if vmap_inverse[nodeID] > 0 ##### if the old node has a new ID in the new graph (it is in the giant component)
                    push!(new_higher_vec[vmap_inverse[nodeID]], (vmap_inverse[simplx[1]], vmap_inverse[simplx[2]]))
                    count_simpl += 1
                end
            end
        end
        count_pairw = 0
        for (nodeID, pairw_4node) in enumerate(pairw_vec)
            for pairw in pairw_4node
                if vmap_inverse[nodeID] > 0 ##### if the old node has a new ID in the new graph (it is in the giant component)
                    push!(new_pairw_vec[vmap_inverse[nodeID]], (vmap_inverse[pairw]))
                    count_pairw += 1
                end
            end
        end
        graph = graph2
        higher_vec = new_higher_vec
        pairw_vec = new_pairw_vec
    end
    return pairw_vec, higher_vec
end

function RR_hypergraph(num_plrs::Int64, kp::Int64, kh::Int64, o::Float32, graph_path::String)
    
    rnd_ind = rand(0:99)
    pairw_data = readdlm(graph_path * "/pairwise_graph_N_$(num_plrs)_kp_$(kp)_kh_$(kh)_o_$(o)_$(rnd_ind).txt")
    higher_data = readdlm(graph_path * "/higher_order_graph_N_$(num_plrs)_kp_$(kp)_kh_$(kh)_o_$(o)_$(rnd_ind).txt")

    # Initialize the vectors
    pairw_vec = [Vector{Int64}() for node in 1:num_plrs]
    higher_vec = [Vector{Tuple{Int64, Int64}}() for node in 1:num_plrs]

    graph_pair = SimpleGraph(num_plrs)
    graph_higher = SimpleGraph(num_plrs)

    # Fill pairw_vec
    for i in 1:num_plrs
        this_node_neighbours = pairw_data[:,i]
        for this_neighbour in this_node_neighbours
            push!(pairw_vec[i], Int(this_neighbour))
            add_edge!(graph_pair, i, Int(this_neighbour))
        end
    end

    # Fill higher_vec
    for i in 1:num_plrs
        this_node_neighbours = higher_data[:,i]
        for j in 1:Int(0.5*length(this_node_neighbours))
            push!(higher_vec[i], (Int(this_node_neighbours[2*j-1]), Int(this_node_neighbours[2*j])))
            add_edge!(graph_higher, i, Int(this_node_neighbours[2*j-1]))
            add_edge!(graph_higher, i, Int(this_node_neighbours[2*j]))
        end
    end

    pair_edges = edges(graph_pair)
    higher_edges = edges(graph_higher)
    pair_triangles = maximal_cliques(graph_pair)
    higher_triangles = maximal_cliques(graph_higher)

    return pairw_vec, higher_vec, pair_edges, higher_edges, pair_triangles, higher_triangles
end
##################################################################



### memory write or update #######################################
function memory_write_double(mem_used::Int64, memory_capacity::Int64,
    memory_pairwise_strat::AbstractArray{Int64,2}, memory_higher_strat::AbstractArray{Int64,2},
    pairwise_strat::Vector{Int64}, higher_strat::Vector{Int64},
    p_overwrite::Float32,
    num_pairwise_coop::Int64, num_higher_coop::Int64)

    if mem_used < memory_capacity
        mem_idx = mem_used + 1
        memory_pairwise_strat[:, mem_idx] = pairwise_strat
        memory_higher_strat[:, mem_idx] = higher_strat
        mem_used += 1
    else
        if rand() < p_overwrite
            ov_mem_ID = rand(1:mem_used)
            memory_pairwise_strat[:, ov_mem_ID] = pairwise_strat
            memory_higher_strat[:, ov_mem_ID] = higher_strat

            num_pairwise_coop = sum(pairwise_strat)
            num_higher_coop = sum(higher_strat)
        end
    end

    return mem_used, memory_pairwise_strat, memory_higher_strat, num_pairwise_coop, num_higher_coop
end
##################################################################


########### payoff entries #############################
function payoff_tensor(Rh::Float32, Sh::Float32, Th::Float32, Ph::Float32, Gh::Float32, Wh::Float32)
    tens = zeros(2, 2, 2)
    tens[:, :, 1] = [Rh Gh; Th Wh]
    tens[:, :, 2] = [Gh Sh; Wh Ph]
    return Float32.(tens)
end

function payoff_matrix(Rp::Float32, Sp::Float32, Tp::Float32, Pp::Float32)
    matr = [Rp Sp; Tp Pp]
    return Float32.(matr)
end
##################################################################


############ evolution of the system ##################
@views function optimized_payoff_calculation_double(
    pairwise_strat::Vector{Int64}, higher_strat::Vector{Int64},
    this_pairw_vec_focal::Vector{Int64}, this_higher_vec_focal::Vector{Tuple{Int64,Int64}},
    this_pairw_vec_model::Vector{Int64}, this_higher_vec_model::Vector{Tuple{Int64,Int64}},
    f_p_strat::Int64, f_h_strat::Int64, 
    m_p_strat::Int64, m_h_strat::Int64,
    payoff_mats::Array{Float32, 2}, payoff_tens::Array{Float32, 3})

    f_payoff = 0.0
    m_payoff = 0.0

    for i in 1:length(this_pairw_vec_focal)
        f_payoff += payoff_mats[2 - f_p_strat, 2 - pairwise_strat[this_pairw_vec_focal[i]]]
    end

    for i in 1:length(this_higher_vec_focal)
        f_payoff += payoff_tens[2 - f_h_strat, 2 - higher_strat[this_higher_vec_focal[i][1]], 2 - higher_strat[this_higher_vec_focal[i][2]]]
    end

    for i in 1:length(this_pairw_vec_model)
        m_payoff += payoff_mats[2 - m_p_strat, 2 - pairwise_strat[this_pairw_vec_model[i]]]
    end

    for i in 1:length(this_higher_vec_model)
        m_payoff += payoff_tens[2 - m_h_strat, 2 - higher_strat[this_higher_vec_model[i][1]], 2 - higher_strat[this_higher_vec_model[i][2]]]
    end


    return f_payoff, m_payoff

end

function evolve_hypergraph_double(
    num_plrs::Int64, num_pairwise_coop::Int64, num_higher_coop::Int64,
    pairwise_strat::Vector{Int64}, higher_strat::Vector{Int64}, wf::Float32,
    pairw_vec::Vector{Vector{Int64}}, higher_vec::Vector{Vector{Tuple{Int64,Int64}}},
    payoff_mats::Array{Float32, 2}, payoff_tens::Array{Float32, 3}, p_switch::Float32)


    #= focal player and model_player =#
    fID = rand(1:num_plrs)
    f_p_strat = pairwise_strat[fID]
    f_h_strat = higher_strat[fID]

    len_pairw_vec = length(pairw_vec[fID])
    len_higher_vec = length(higher_vec[fID])

    r1 = rand()
    if len_pairw_vec == 0
        mID = rand(rand(higher_vec[fID]))
    elseif len_higher_vec == 0
        mID = rand(pairw_vec[fID])
    else
        if r1 < len_pairw_vec / (len_pairw_vec + 2*len_higher_vec)
            mID = rand(pairw_vec[fID])
        else
            mID = rand(rand(higher_vec[fID]))
        end
    end

    m_p_strat = pairwise_strat[mID]
    m_h_strat = higher_strat[mID]

    r1 = rand()
    if (r1 < 0.5) && (f_p_strat == m_p_strat)
        return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat
    elseif (r1 >= 0.5) && (f_h_strat == m_h_strat)
        return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat
    else
        #= payoff calculation =#
        f_payoff, m_payoff = optimized_payoff_calculation_double(
            pairwise_strat, higher_strat,
            pairw_vec[fID], higher_vec[fID],
            pairw_vec[mID], higher_vec[mID],
            f_p_strat, f_h_strat,
            m_p_strat, m_h_strat,
            payoff_mats, payoff_tens)
            

        trans_prob = 1.0 / (1.0 + exp(-wf*(m_payoff - f_payoff)))

        if rand() < trans_prob
            if r1 < 0.5
                if rand() < p_switch
                    pairwise_strat[fID] = m_h_strat
                    num_pairwise_coop += m_h_strat - f_p_strat
                else
                    pairwise_strat[fID] = m_p_strat
                    num_pairwise_coop += m_p_strat - f_p_strat
                end
            else
                if rand() < p_switch
                    higher_strat[fID] = m_p_strat
                    num_higher_coop += m_p_strat - f_h_strat
                else
                    higher_strat[fID] = m_h_strat
                    num_higher_coop += m_h_strat - f_h_strat
                end
            end
        end

        return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat
    end 
end




function evolve_hypergraph_coherence(
    num_plrs::Int64, num_pairwise_coop::Int64, num_higher_coop::Int64,
    pairwise_strat::Vector{Int64}, higher_strat::Vector{Int64}, wf::Float32,
    pairw_vec::Vector{Vector{Int64}}, higher_vec::Vector{Vector{Tuple{Int64,Int64}}},
    payoff_mats::Array{Float32, 2}, payoff_tens::Array{Float32, 3}, p_switch::Float32)


    #= focal player and model_player =#
    fID = rand(1:num_plrs)
    f_p_strat = pairwise_strat[fID]
    f_h_strat = higher_strat[fID]

    len_pairw_vec = length(pairw_vec[fID])
    len_higher_vec = length(higher_vec[fID])

    r1 = rand()
    if len_pairw_vec == 0
        mID = rand(rand(higher_vec[fID]))
    elseif len_higher_vec == 0
        mID = rand(pairw_vec[fID])
    else
        if r1 < len_pairw_vec / (len_pairw_vec + 2*len_higher_vec)
            mID = rand(pairw_vec[fID])
        else
            mID = rand(rand(higher_vec[fID]))
        end
    end

    m_p_strat = pairwise_strat[mID]
    m_h_strat = higher_strat[mID]

    r1 = rand()
    if (r1 < 0.5) && (f_p_strat == m_p_strat)
        return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat
    elseif (r1 >= 0.5) && (f_h_strat == m_h_strat)
        return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat
    else

        if rand() < p_switch
            
            if r1 < 0.5
                pairwise_strat[fID] = f_h_strat
                num_pairwise_coop += f_h_strat - f_p_strat
            else
                higher_strat[fID] = f_p_strat
                num_higher_coop += f_p_strat - f_h_strat
            end
        
        else

            #= payoff calculation =#
            f_payoff, m_payoff = optimized_payoff_calculation_double(
                pairwise_strat, higher_strat,
                pairw_vec[fID], higher_vec[fID],
                pairw_vec[mID], higher_vec[mID],
                f_p_strat, f_h_strat,
                m_p_strat, m_h_strat,
                payoff_mats, payoff_tens)
                

            trans_prob = 1.0 / (1.0 + exp(-wf*(m_payoff - f_payoff)))

            if rand() < trans_prob
                if r1 < 0.5
                    pairwise_strat[fID] = m_p_strat
                    num_pairwise_coop += m_p_strat - f_p_strat
                else
                    higher_strat[fID] = m_h_strat
                    num_higher_coop += m_h_strat - f_h_strat
                end
            end
        end

        return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat
    end 
end
##################################################################



### teleport the system
function time_machine_double(mem_used :: Int64, 
    memory_pairwise_strat::AbstractArray{Int64,2}, memory_higher_strat::AbstractArray{Int64,2})


    slot_ID = rand(1:mem_used)
    pairwise_strat = view(memory_pairwise_strat, :, slot_ID)
    higher_strat = view(memory_higher_strat, :, slot_ID)

    num_pairwise_coop = sum(pairwise_strat)
    num_higher_coop = sum(higher_strat)

    return num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat

end
##################################################################


end # module
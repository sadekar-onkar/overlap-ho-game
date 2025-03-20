parent_dir = dirname(pwd())

src_path = parent_dir * "/src/"
dat_path = parent_dir * "/data/final_data"
graph_path = parent_dir * "/data/graph_data"

include(src_path * "HO_PRL_monolayer.jl")

using Revise
using BenchmarkTools
using JLD2
using Plots
using LaTeXStrings
using Random
using Statistics
using DelimitedFiles
using TimerOutputs


function main()

    #= initialize everything =#
    _strat::Vector{Int64} = HO_PRL_payoff.init_plrs_strats_fixnumber(num_plrs, rhoi)

    pairw_vec, higher_vec, pair_edges, higher_edges, pair_triangles, higher_triangles = HO_PRL_payoff.RR_hypergraph(num_plrs, kp, kh, overlap, graph_path)


    #= declare data arrays =#
    temporal_coop_single_run = zeros(Int64, tot_sim_length)


    ###### Initializing the memory
    mem_used = 0   ####how many memory slots have been used
    mem_capacity = 10_000
    p_overwrite  = Float32(0.01)   ####prob of overwrite a state in memory
    memory_strat = zeros(Int64, num_plrs, mem_capacity)

    num_coop = sum(_strat)

    #= evolve the system till tot_sim_length =#
    for step in 1:tot_sim_length
        for player_step in 1:num_plrs
            
            ##### memory write or update
            mem_used, memory_strat, num_coop = HO_PRL_payoff.memory_write_single(
                mem_used, mem_capacity, memory_strat, 
                _strat, p_overwrite, num_coop)

            ##### evolve the state of the system
            num_coop, _strat = HO_PRL_payoff.evolve_hypergraph_single(
                num_plrs, num_coop, _strat, wf,
                pairw_vec, higher_vec, 
                payoff_tens, payoff_mats)

            ###### teleporation step
            if (num_coop == 0) || (num_coop == num_plrs) 
                num_coop, _strat = HO_PRL_payoff.time_machine_single(mem_used, memory_strat)
            end

        end # player_step
        temporal_coop_single_run[step] = num_coop
    end # step


    #= save the data =#
    filename = "N_$(num_plrs)_overlap_$(overlap)_kp_$(kp)_kh_$(kh)_Rp_$(Rp)_Sp_$(Sp)_Tp_$(Tp)_Pp_$(Pp)_Gh_$(Gh)_Wh_$(Wh)_wf_$(round(wf, digits=3))_rhoi_$(rhoi)_runID_$(runID)"


    writedlm(dat_path * "/RR_monolayer_temporal_coop_single_run_$(filename).txt", temporal_coop_single_run)
    
end # main


# Initialize the configuration struct
const overlap = parse(Float32, ARGS[1])
const Gh = parse(Float32, ARGS[2])
const rhoi = parse(Float32, ARGS[3])
const runID = parse(Int64, ARGS[4])


const num_plrs = 1500
const kp, kh = 4, 2

const Rp, Sp, Tp, Pp = (Float32(1.0), Float32(-0.1), Float32(1.1), Float32(0.0))
const Rh, Sh, Th, Ph = (Float32(2*Rp), Float32(2*Sp), Float32(2*Tp), Float32(2*Pp))
const Wh = Float32(0.7)

const tot_sim_length = Int64(1e5)

const payoff_tens = HO_PRL_payoff.payoff_tensor(Rh, Sh, Th, Ph, Gh, Wh)
const payoff_mats = HO_PRL_payoff.payoff_matrix(Rp, Sp, Tp, Pp)
const wf = Float32(1.0/(kp+kh))


main()
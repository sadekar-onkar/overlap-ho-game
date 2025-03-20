parent_dir = dirname(pwd())

src_path = parent_dir * "/src/"
dat_path = parent_dir * "/data/final_data"
graph_path = parent_dir * "/data/graph_data"

include(src_path * "HO_PRL_bilayer.jl")


######################## CHECK THE END PARAMETER ALWAYS ############################
####################################################################################
####################################################################################
####################################################################################
####################################################################################


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
    pairwise_strat::Vector{Int64} = HO_PRL_strategy.init_plrs_strats_fixnumber(num_plrs, rho_pi)
    higher_strat::Vector{Int64} = HO_PRL_strategy.init_plrs_strats_fixnumber(num_plrs, rho_hi)

    # pairw_vec, higher_vec = HO_PRL_strategy.ER_hypergraph(num_plrs, ktot, delta)
    pairw_vec, higher_vec, pair_edges, higher_edges, pair_triangles, higher_triangles = HO_PRL_strategy.RR_hypergraph(num_plrs, kp, kh, overlap, graph_path)

    #= declare data arrays =#
    temporal_coop_pairwise_single_run = zeros(Int64, stationary_time)
    temporal_coop_higher_single_run = zeros(Int64, stationary_time)


    ###### Initializing the memory
    mem_used = 0   ####how many memory slots have been used
    mem_capacity = 10_000
    p_overwrite  = Float32(0.01)   ####prob of overwrite a state in memory
    memory_pairwise_strat = zeros(Int64, num_plrs, mem_capacity)
    memory_higher_strat = zeros(Int64, num_plrs, mem_capacity)

    num_pairwise_coop = sum(pairwise_strat)
    num_higher_coop = sum(higher_strat)


    #= evolve the system till therm_time =#
    for step in 1:therm_time
        for player_step in 1:2*num_plrs
            # println(step)
            
            ##### memory write or update
            mem_used, memory_pairwise_strat, memory_higher_strat, num_pairwise_coop, num_higher_coop = HO_PRL_strategy.memory_write_double(
                mem_used, mem_capacity, 
                memory_pairwise_strat, memory_higher_strat, 
                pairwise_strat, higher_strat,
                p_overwrite, num_pairwise_coop, num_higher_coop)


            ##### evolve the state of the system
            num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat = HO_PRL_strategy.evolve_hypergraph_double(
                num_plrs, num_pairwise_coop, num_higher_coop, 
                pairwise_strat, higher_strat, wf,
                pairw_vec, higher_vec, 
                payoff_mats, payoff_tens, p_switch)


            ###### teleporation step
            if (num_pairwise_coop == 0) || (num_higher_coop == 0) || (num_pairwise_coop == num_plrs) || (num_higher_coop == num_plrs)
                num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat = HO_PRL_strategy.time_machine_double(mem_used, memory_pairwise_strat, memory_higher_strat)
            end
        end # player_step
    end # step


    

    #= save the stationary state data =#
    for step in 1:stationary_time
        for player_step in 1:2*num_plrs
            ##### memory write or update
            mem_used, memory_pairwise_strat, memory_higher_strat, num_pairwise_coop, num_higher_coop = HO_PRL_strategy.memory_write_double(
                mem_used, mem_capacity, 
                memory_pairwise_strat, memory_higher_strat, 
                pairwise_strat, higher_strat,
                p_overwrite, num_pairwise_coop, num_higher_coop)

            ##### evolve the state of the system

            num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat = HO_PRL_strategy.evolve_hypergraph_double(
                num_plrs, num_pairwise_coop, num_higher_coop, 
                pairwise_strat, higher_strat, wf,
                pairw_vec, higher_vec, 
                payoff_mats, payoff_tens, p_switch)

            ###### teleporation step
            if (num_pairwise_coop == 0) || (num_higher_coop == 0) || (num_pairwise_coop == num_plrs) || (num_higher_coop == num_plrs)
                num_pairwise_coop, num_higher_coop, pairwise_strat, higher_strat = HO_PRL_strategy.time_machine_double(mem_used, memory_pairwise_strat, memory_higher_strat)
            end
        end # player_step
        temporal_coop_pairwise_single_run[step] = num_pairwise_coop
        temporal_coop_higher_single_run[step] = num_higher_coop
    end # step


    #= save the data =#
    filename = "N_$(num_plrs)_kp_$(kp)_kh_$(kh)_overlap_$(overlap)_Rp_$(Rp)_Sp_$(Sp)_Tp_$(Tp)_Pp_$(Pp)_Gh_$(Gh)_Wh_$(Wh)_wf_$(round(wf, digits=3))_rho_pi_$(rho_pi)_rho_hi_$(rho_hi)_p_switch_$(p_switch)_runID_$(runID)"

    writedlm(dat_path * "/RR-temporal_coop_pairwise_single_run_$(filename).txt", temporal_coop_pairwise_single_run)
    writedlm(dat_path * "/RR-temporal_coop_higher_single_run_$(filename).txt", temporal_coop_higher_single_run)
end # main



# Initialize the configuration struct
const overlap = parse(Float32, ARGS[1])
const Gh = parse(Float32, ARGS[2])
const rho_pi = parse(Float32, ARGS[3])
const rho_hi = parse(Float32, ARGS[4])
const p_switch = parse(Float32, ARGS[5])
const runID = parse(Int64, ARGS[6])


const num_plrs = 1500
const kp, kh = 4, 2

const Rp, Sp, Tp, Pp = (Float32(1.0), Float32(-0.1), Float32(1.1), Float32(0.0))
const Rh, Sh, Th, Ph = (Float32(2*Rp), Float32(2*Sp), Float32(2*Tp), Float32(2*Pp))
const Wh = Float32(0.7)

const therm_time = Int64(9e4)
const stationary_time = Int64(1e4)

const payoff_tens = HO_PRL_strategy.payoff_tensor(Rh, Sh, Th, Ph, Gh, Wh)
const payoff_mats = HO_PRL_strategy.payoff_matrix(Rp, Sp, Tp, Pp)
const wf = Float32(1.0/(kp+kh))

main()
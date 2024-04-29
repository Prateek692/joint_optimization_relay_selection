import numpy as np
import pandas as pd
import math
import sys
import random
from ipynb.fs.full.Formulas import *
from ipynb.fs.full.Environment import *
from ipynb.fs.full.Flow_Grouping_Algo import *




def concurrent_transmission():
    groups = generate_groups()
    system_throughput = 0
    total_number_of_slots = 0
    rsi = calculate_rsi()
    
    for group in groups:
        uavs_in_group = []
        
        for flow_index in group:
            relay_type, candidate_set, random_candidate = flows_candidate_relay_set_pairs[str([flow_index,0.0])]
            
            if random_candidate != -1 and random_candidate.type == 'uav':
                uavs_in_group.append(uav_object_to_index_map[random_candidate])
                uav_object_to_index_map[random_candidate] = len(uavs_in_group) - 1
        
        current_group_slots = 0
        new_hops_completed = False
        completed_sender_vehicles = set()
        completed_receiver_vehicles = set()
#         completed_sender_uavs = set()
#         completed_receiver_uavs = set()
        group = list(group)
        group_size = len(group)
        flows_remaining = group_size
        uav_size = len(uavs_in_group)
        current_hops = np.ones(group_size)
        max_hops = np.ones(group_size)
        bits_in_each_flow = np.zeros(group_size)
        data_rate_of_each_flow_v2v = np.zeros(group_size)
#         denom_of_each_flow_v2v = np.zeros(group_size)
        data_rate_of_each_flow_u2v = np.zeros((uav_size, group_size))
#         denom_of_each_flow_u2v = np.zeros(group_size)
        data_rate_of_each_flow_v2u = np.zeros((group_size, uav_size))
#         denom_of_each_flow_v2u = np.zeros(group_size)
        a_v2v = np.ones((group_size, group_size))
        b_v2v = np.ones((group_size, group_size))
        a_u2v = np.ones((group_size, group_size))
        b_u2v = np.ones((group_size, group_size))
        ku = (wavelength / (4 * math.pi)) ** pl_factor_for_u2v
        kv = (wavelength / (4 * math.pi)) ** pl_factor_for_v2v
        
        for group_index in range(group_size):
            flow_index = group[group_index]
            system_throughput += throughputs[flow_index]
            relay_type, candidate_set, random_candidate = flows_candidate_relay_set_pairs[str([flow_index,0.0])]

            if relay_type != 'dont_relay' and random_candidate != -1:
                max_hops[group_index] = 2
        
        # print('N0W:', background_noise * system_bandwidth)
        # print('RSI:', rsi)

        #v2v
        # print('v2v num denom pairs')
        for group_index_i in range(group_size):
            flow_index_i = group[group_index_i]
            sender_i = flows[flow_index_i][0]
            receiver_i = flows[flow_index_i][1]
            distance_i_i = euclidean_distance(*sender_i.calculate_position(total_number_of_slots * slot_duration), *receiver_i.calculate_position(total_number_of_slots * slot_duration))
            num = calculate_received_signal_power_direct_link(distance_i_i)
            denom = background_noise * system_bandwidth
            
            for group_index_j in range(group_size):
                if group_index_i != group_index_j:
                    flow_index_j = group[group_index_j]
                    sender_j = flows[flow_index_j][0]
                    distance_j_i = euclidean_distance(*sender_j.calculate_position(total_number_of_slots * slot_duration), *receiver_i.calculate_position(total_number_of_slots * slot_duration))
                    denom += a_v2v[group_index_j, group_index_i] * calculate_mutual_interference(distance_j_i) + b_v2v[group_index_j, group_index_i] * rsi 
            # print(num,denom)
            data_rate_of_each_flow_v2v[group_index_i] = calculate_data_rate(num, denom)
            
        #u2v
        # print('u2v num denom pairs')
        for group_index_u in range(uav_size):
            uav_index_u = uavs_in_group[group_index_u]
            sender_u = list_of_uavs[uav_index_u]
            
            for group_index_k in range(group_size):
                flow_index_k = group[group_index_k]
                receiver_k = flows[flow_index_k][1]
                distance_u_k = math.sqrt(euclidean_distance(*sender_u.calculate_position(total_number_of_slots * slot_duration), *receiver_k.calculate_position(total_number_of_slots * slot_duration)) ** 2 + height_of_uav ** 2)
                num = ku * transmit_power_uav * maximum_antenna_gain * (distance_u_k ** -pl_factor_for_u2v) * calculate_small_scale_power_fading()
                denom = background_noise * system_bandwidth
                # print('u2v',num)
                for group_index_w in range(group_size):
                    if group_index_k != group_index_w:
                        flow_index_w = group[group_index_w]
                        sender_w = flows[flow_index_w][0]
                        distance_w_k = euclidean_distance(*sender_w.calculate_position(total_number_of_slots * slot_duration), *receiver_k.calculate_position(total_number_of_slots * slot_duration))
                        denom += a_u2v[group_index_w, group_index_k] * calculate_mutual_interference(distance_w_k) + b_u2v[group_index_w, group_index_k] * rsi 
                        print(calculate_mutual_interference(distance_w_k))
                # print(num,denom)
                data_rate_of_each_flow_u2v[group_index_u, group_index_k] = calculate_data_rate(num, denom)
                
        #v2u
        # print('v2u num denom pairs')
        for group_index_k in range(group_size):
            flow_index_k = group[group_index_k]
            sender_k = flows[flow_index_k][0]
            
            for uav_index_u in range(uav_size):
                uav_index_u = uavs_in_group[group_index_u]
                receiver_u = list_of_uavs[uav_index_u]
                distance_k_u = math.sqrt(euclidean_distance(*sender_k.calculate_position(total_number_of_slots * slot_duration), *receiver_u.calculate_position(total_number_of_slots * slot_duration)) ** 2 + height_of_uav ** 2)
                num = kv * transmission_power * maximum_antenna_gain * (distance_k_u ** -pl_factor_for_v2v) * calculate_small_scale_power_fading()
                denom = background_noise * system_bandwidth
                # print(num,denom)
                data_rate_of_each_flow_v2u[group_index_k, group_index_u] = calculate_data_rate(num, denom)

        # print('Initial:')
        # print(data_rate_of_each_flow_u2v)
        # print(data_rate_of_each_flow_v2u)
        # print(data_rate_of_each_flow_v2v)
        # print('u2v:',np.min(data_rate_of_each_flow_u2v),np.max(data_rate_of_each_flow_u2v))
        # print('v2u:',np.min(data_rate_of_each_flow_v2u),np.max(data_rate_of_each_flow_v2u))
        # print('v2v:',np.min(data_rate_of_each_flow_v2v),np.max(data_rate_of_each_flow_v2v))

        while flows_remaining:
            # print(flows_remaining, current_group_slots)
            if new_hops_completed:
                new_hops_completed = False
                
                #v2v
#                 data_rate_of_each_flow_v2v = np.zeros(group_size)
                
                for group_index_i in range(group_size):
                    if group_index_i in completed_receiver_vehicles:
                        continue
                        
                    flow_index_i = group[group_index_i]
                    sender_i = flows[flow_index_i][0]
                    receiver_i = flows[flow_index_i][1]
                    distance_i_i = euclidean_distance(*sender_i.calculate_position((total_number_of_slots + current_group_slots) * slot_duration),*receiver_i.calculate_position((total_number_of_slots + current_group_slots) * slot_duration))
                    num = calculate_received_signal_power_direct_link(distance_i_i)
                    denom = background_noise * system_bandwidth

                    for group_index_j in range(group_size):
                        if group_index_i == group_index_j or group_index_j in completed_sender_vehicles:
                            continue
                            
                        flow_index_j = group[group_index_j]
                        sender_j = flows[flow_index_j][0]
                        distance_j_i = euclidean_distance(*sender_j.calculate_position((total_number_of_slots + current_group_slots) * slot_duration), *receiver_i.calculate_position((total_number_of_slots + current_group_slots) * slot_duration))
                        denom += a_v2v[group_index_j, group_index_i] * calculate_mutual_interference(distance_j_i) + b_v2v[group_index_j, group_index_i] * rsi 
                    
                    data_rate_of_each_flow_v2v[group_index_i] = calculate_data_rate(num, denom)
                    
                #u2v
#                 data_rate_of_each_flow_u2v = np.zeros((uav_size, group_size))
                
                for group_index_u in range(uav_size):
                    uav_index_u = uavs_in_group[group_index_u]
                    sender_u = list_of_uavs[uav_index_u]

                    for group_index_k in range(group_size):
                        if group_index_k in completed_receiver_vehicles:
                            continue
                            
                        flow_index_k = group[group_index_k]
                        receiver_k = flows[flow_index_k][1]
                        distance_u_k = math.sqrt(euclidean_distance(*sender_u.calculate_position((total_number_of_slots + current_group_slots) * slot_duration), *receiver_k.calculate_position((total_number_of_slots + current_group_slots) * slot_duration)) ** 2 + height_of_uav ** 2)
                        num = ku * transmit_power_uav * maximum_antenna_gain * (distance_u_k ** -pl_factor_for_u2v) * calculate_small_scale_power_fading()
                        denom = background_noise * system_bandwidth

                        for group_index_w in range(group_size):
                            if group_index_k == group_index_w or group_index_w in completed_sender_vehicles:
                                continue
                                
                            flow_index_w = group[group_index_w]
                            sender_w = flows[flow_index_w][0]
                            distance_w_k = euclidean_distance(*sender_w.calculate_position((total_number_of_slots + current_group_slots) * slot_duration), *receiver_k.calculate_position((total_number_of_slots + current_group_slots) * slot_duration))
                            denom += a_u2v[group_index_w, group_index_k] * calculate_mutual_interference(distance_w_k) + b_u2v[group_index_w, group_index_k] * rsi 

                        data_rate_of_each_flow_u2v[group_index_u, group_index_k] = calculate_data_rate(num, denom)
                
                #v2u
#                 data_rate_of_each_flow_v2u = np.zeros((group_size, uav_size))
            
                for group_index_k in range(group_size):
                    if group_index_k in completed_sender_vehicles:
                        continue
                        
                    flow_index_k = group[group_index_k]
                    sender_k = flows[flow_index_k][0]

                    for uav_index_u in range(uav_size):
                        uav_index_u = uavs_in_group[group_index_u]
                        receiver_u = list_of_uavs[uav_index_u]
                        distance_k_u = math.sqrt(euclidean_distance(*sender_k.calculate_position(total_number_of_slots * slot_duration), *receiver_u.calculate_position(total_number_of_slots * slot_duration)) ** 2 + height_of_uav ** 2)
                        num = kv * transmission_power * maximum_antenna_gain * (distance_k_u ** -pl_factor_for_v2v) * calculate_small_scale_power_fading()
                        denom = background_noise * system_bandwidth
                        data_rate_of_each_flow_v2u[group_index_k, group_index_u] = calculate_data_rate(num, denom)
                
            current_group_slots += 1
            # print(current_group_slots,':')
            # print(data_rate_of_each_flow_u2v)
            # print(data_rate_of_each_flow_v2u)
            # print(data_rate_of_each_flow_v2v)
            



            
            for group_index in range(group_size):
                if current_hops[group_index] > max_hops[group_index]:
                    continue
                
                flow_index = group[group_index]
                
                if max_hops[group_index] == 2:
                    relay_type, candidate_set, random_candidate = flows_candidate_relay_set_pairs[str([flow_index,0.0])]
                    if random_candidate.type == 'vehicle':
                        
                        if data_rate_of_each_flow_v2v[group_index] * current_group_slots >= throughputs[flow_index] * number_of_time_slots:
                            # Idhar nahi ho raha call
                            completed_sender_vehicles.add(group_index)
                            completed_receiver_vehicles.add(group_index)
                            current_hops[group_index] += 2
                            new_hops_completed = True
                            flows_remaining -= 1
                    else:
                        uav_index = uav_object_to_index_map[random_candidate]

                        if current_hops[group_index] == 1:
                            if data_rate_of_each_flow_v2u[group_index, uav_index] * current_group_slots >= throughputs[flow_index] * number_of_time_slots:
                                completed_sender_vehicles.add(group_index)
#                                 completed_receiver_uavs(uav_index)
                                current_hops[group_index] += 1
                                new_hops_completed = True
                        else:
                            if (data_rate_of_each_flow_v2u[group_index, uav_index] + data_rate_of_each_flow_u2v[uav_index, group_index]) * current_group_slots  >= throughputs[flow_index] * number_of_time_slots:
#                                 completed_sender_uavs(uav_index)
                                completed_receiver_vehicles.add(group_index)
                                current_hops[group_index] += 1
                                new_hops_completed = True
                                flows_remaining -= 1
                else:
                    if data_rate_of_each_flow_v2v[group_index] * current_group_slots  >= throughputs[flow_index] * number_of_time_slots:
                        completed_sender_vehicles.add(group_index)
                        completed_receiver_vehicles.add(group_index)
                        current_hops[group_index] += 1    
                        new_hops_completed = True
                        flows_remaining -= 1
            
            if not new_hops_completed:
                continue
            
#             all_hops_completed = True
            
#             for group_index in range(group_size):
#                 if current_hops[group_index] <= max_hops[group_index]:
#                     all_hops_completed = False
#                     break
            
#             if all_hops_completed:
#                 break
                
        total_number_of_slots += current_group_slots 
        
    system_throughput /= (total_number_of_slots * 0.1)
    
    return total_number_of_slots, system_throughput


concurrent_transmission()

import numpy as np
import heapq
from queue import Queue
import random
import math
import pandas as pd

class Event:
    def __init__(self, ev_type, id, start):
        self.type = ev_type
        self.owner = id
        self.start_time = start

class Customer:
    def __init__(self, arrival_num, arrival, service, delay, queue):
        self.id = arrival_num
        self.arrival_time = arrival
        self.waiting_time = 0
        self.service_time = service
        self.max_waiting_time = delay
        self.selected_queue = queue

def simulate(N, lamda, teta, delay_type, mu_1, mu_2, K1, K2):

    MAX_ARRIVAL_NUM = N
    service_rate = [mu_1, mu_2]
    arrival_rate = lamda
    delay_rate = teta

    ARRIVAL_TYPE = "ARRIVAL"
    DEPARTURE_TYPE = "DEPARTURE"
    LEAVING_TYPE = "LEAVE"

    FIXED_DELAY_TYPE = "FIX"
    EXP_DELAY_TYPE = "EXP"

    events = list()
    q1 = list()
    q2 = list()
    Queues = [0, q1, q2]    
    Customers = list()

    #First Arrival 
    arrival_time = 1 / arrival_rate
    service_time = 1 / service_rate[0]
    delay_time = delay_rate
    new_customer = Customer(0, arrival_time, service_time, delay_time, 1)
    Customers.append(new_customer)
    arrival_event = Event(ARRIVAL_TYPE, 0, 0)
    heapq.heappush(events, (arrival_time, arrival_event))

    Time = 0
    Num = 0
    prev_time = 0
    Blocked = 0
    Delayed = 0

    while (len(events) > 0 and len(Customers) < MAX_ARRIVAL_NUM) :
        #Poping next event
        ev_time, event = heapq.heappop(events)
        Time = event.start_time
        arrival_num = event.owner
        ev_type = event.type
        customer = Customers[arrival_num]

        #Calculating Number of customers in queue
        Num += (Time - prev_time) * (len(Queues[1]) + len(Queues[2]))
        prev_time = Time
        
        if ev_type == ARRIVAL_TYPE:
            
            #Scheduling next arrival
            arrival_time = Time + (-1 / arrival_rate) * math.log(1 - random.uniform(0, 1))
            service_time = (-1 / service_rate[0]) * math.log(1 - random.uniform(0, 1))
            if delay_type == EXP_DELAY_TYPE :
                delay_time = (-1 * delay_rate) * math.log(1 - random.uniform(0, 1))
            elif delay_type == FIXED_DELAY_TYPE :
                delay_time = delay_rate
            new_customer = Customer(len(Customers), arrival_time, service_time, delay_time, 1)
            Customers.append(new_customer)
            arrival_event = Event(ARRIVAL_TYPE, new_customer.id, arrival_time)
            heapq.heappush(events, (arrival_time, arrival_event))

            #Adding current customer to queue
            if (len(Queues[1]) == K1 and len(Queues[2]) == K2) :
                Blocked += 1
            else :
                coin = np.random.uniform(0, 1)
                if coin < 0.5 :
                    if len(Queues[2]) < K2 :
                        heapq.heappush(Queues[2], (customer.arrival_time, customer))
                        customer.selected_queue = 2
                    else :
                        heapq.heappush(Queues[1], (customer.arrival_time, customer))
                        customer.selected_queue = 1
                else :
                    if len(Queues[1]) < K1 :
                        heapq.heappush(Queues[1], (customer.arrival_time, customer))
                        customer.selected_queue = 1
                    else :
                        heapq.heappush(Queues[2], (customer.arrival_time, customer))
                        customer.selected_queue = 2
                #Changing the customer's service time for different server
                if customer.selected_queue == 2:
                    new_service_time = (-1 / service_rate[1]) * math.log(1 - random.uniform(0, 1))
                    customer.service_time = new_service_time
                #Checking emptiness of queue and adding departure event
                if len(Queues[customer.selected_queue]) == 1 :
                    departure_time = Time + customer.service_time
                    departure_event = Event(DEPARTURE_TYPE, customer.id, departure_time)
                    heapq.heappush(events, (departure_time, departure_event))
                else :
                #Adding Delay event    
                    leaving_time = Time + customer.max_waiting_time
                    leaving_event = Event(LEAVING_TYPE, customer.id, leaving_time)
                    heapq.heappush(events, (leaving_time, leaving_event))
        
        if ev_type == LEAVING_TYPE:
            #Removing departure event of the leaving customer
            for element in Queues[customer.selected_queue] :
                X, Y = element
                if Y.id == customer.id :
                    Queues[customer.selected_queue].remove(element)
                    break
            Delayed += 1

        if ev_type == DEPARTURE_TYPE:
            #Popping the customer from queue            
            X, old_customer = heapq.heappop(Queues[customer.selected_queue])

            #Assigning new customer to server
            if len(Queues[customer.selected_queue]) > 0 :
                X, new_customer = heapq.heappop(Queues[customer.selected_queue])
                heapq.heappush(Queues[customer.selected_queue], (X, new_customer))
                #Removing Leave event of the departing customer
                for element in events:
                    ev_time, new_event = element
                    if new_event.owner == new_customer.id :
                        events.remove(element)
                        break
                #Adding departure event
                departure_time = Time + new_customer.service_time
                departure_event = Event(DEPARTURE_TYPE, new_customer.id, departure_time)
                heapq.heappush(events, (departure_time, departure_event))

    #Calculating the Pb, Pd, Nc
    Pb_sim = Blocked / MAX_ARRIVAL_NUM
    Pd_sim = Delayed / MAX_ARRIVAL_NUM
    Nc_sim = Num / Time

    return ([Pb_sim, Pd_sim, Nc_sim])

def make_row(lamda, sim_results, anal_results): #Make a dict from results to make an excel file at the end
    Pb_sim, Pd_sim, Nc_sim = sim_results
    Pb_anal, Pd_anal, Nc_anal = anal_results
    row = {
        'lambda' : lamda,
        'Pb_sim' : Pb_sim,
        'Pb_anal' : Pb_anal,
        'Pd_sim' : Pd_sim,
        'Pd_anal' : Pd_anal,
        'Nc_sim' : Nc_sim,
        'Nc_anal' : Nc_anal,   
        }
    return row

def analytic(lamda, teta, delay_type, mu, k):
    phi = [0] * (k+1)
    P_factor = [0] * (k+1)
    sigma = [0] * (k+1)
    sigma[0] = 1
    P_factor[0] = 1
    P = [0] * (k+1)
    sigma_P_factor = 1
    in_sigma = [0] * (k+1)
    in_sigma[0] = 1
    for i in range(1, k+1, 1):
        in_sigma[i] = in_sigma[i - 1] * (mu * teta) / (i)
        sigma[i] = sigma[i - 1] + in_sigma[i]

    if delay_type == "EXP":
        phi[0] = (1 / mu)
    elif delay_type == "FIX":
        phi[1] = (1 / pow(mu, 2)) * (1 - pow(math.e, -1 * mu * teta))

    for i in range(1, k+1, 1):
        if delay_type == "FIX" :
            if i != 1:
                phi[i] = (math.factorial(i) / (pow(mu, i + 1))) * (1 - pow(math.e, -1 * mu * teta) * sigma[i-1])       
        elif delay_type == "EXP" :
            phi[i] = ((i) / (mu + (i / teta))) * phi[i - 1]
    
    P_factor[1] = lamda / mu
    sigma_P_factor += P_factor[1]
    for i in range(2, k+1, 1):
        P_factor[i] = (pow(lamda, i) * phi[i - 1]) / (math.factorial(i - 1))
        sigma_P_factor += P_factor[i]    
    P[0] = 1 / (sigma_P_factor)
    for i in range(1, k+1, 1):
        P[i] = P_factor[i] * P[0]

    Pb_anal= P[k]

    sum = 0
    for i in range(1, k+1, 1):
        sum += P[i]
    print(sum)
    Pd_anal = 1 - P[k] - (mu / lamda) * sum

    Nc_anal = 0
    for i in range(1, k+1, 1):
        Nc_anal += i * P[i]

    return [Pb_anal, Pd_anal, Nc_anal]

if __name__ == '__main__':
    

    #simulation_res = simulate(N= 1_00_000, lamda= 100/10, teta= 3, delay_type= "EXP", mu_1= 1, mu_2= 1, K1= 0, K2= 14)
    #analytic_res = analytic(10, 3, "EXP", 1, 14)
    
    my_list = []
    for x in range (1, 201, 1):
        simulation_res = simulate(N= 1_000_000, lamda= x/10, teta= 3, delay_type= "FIX", mu_1= 1, mu_2= 1, K1= 16, K2= 0)
        analytic_res = analytic(lamda= x/10, teta= 3, delay_type= "FIX", mu= 1, k= 16)
        row = make_row(lamda= x/10, sim_results= simulation_res, anal_results= analytic_res)
        my_list.append(row)
    new_list = pd.DataFrame(my_list)
    new_list.to_excel("FCFS-TetaFix-K16.xlsx")    

    my_list = []
    for x in range (1, 201, 1):
        simulation_res = simulate(N= 1_000_000, lamda= x/10, teta= 3, delay_type= "FIX", mu_1= 1, mu_2= 1, K1= 14, K2= 0)
        analytic_res = analytic(lamda= x/10, teta= 3, delay_type= "FIX", mu= 1, k= 14)
        row = make_row(lamda= x/10, sim_results= simulation_res, anal_results= analytic_res)
        my_list.append(row)
    new_list = pd.DataFrame(my_list)
    new_list.to_excel("FCFS-TetaFix-K14.xlsx")    

    my_list = []
    for x in range (1, 201, 1):
        simulation_res = simulate(N= 1_000_000, lamda= x/10, teta= 3, delay_type= "EXP", mu_1= 1, mu_2= 1, K1= 16, K2= 0)
        analytic_res = analytic(lamda= x/10, teta= 3, delay_type= "EXP", mu= 1, k= 16)
        row = make_row(lamda= x/10, sim_results= simulation_res, anal_results= analytic_res)
        my_list.append(row)
    new_list = pd.DataFrame(my_list)
    new_list.to_excel("FCFS-TetaExp-K16.xlsx")    

    my_list = []
    for x in range (1, 201, 1):
        simulation_res = simulate(N= 1_000_000, lamda= x/10, teta= 3, delay_type= "EXP", mu_1= 1, mu_2= 1, K1= 14, K2= 0)
        analytic_res = analytic(lamda= x/10, teta= 3, delay_type= "EXP", mu= 1, k= 14)
        row = make_row(lamda= x/10, sim_results= simulation_res, anal_results= analytic_res)
        my_list.append(row)
    new_list = pd.DataFrame(my_list)
    new_list.to_excel("FCFS-TetaExp-K14.xlsx")    


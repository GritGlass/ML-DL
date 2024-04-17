Order Batching Optimization for Warehouses with Clusrer-Picking   
Aaya Aboelfotoh etc.
===================================================================

## 25th International Conference on Production Research Manufacuring Innovation: Cyber Physical Manufacturing
## August 9-14, 2019, Chicago, Illinois(USA)

### 1. Problem
![Alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/problem_img.png)   


### 2. Method
#### A. First Come First Serve(FCFS)
the order number also corresponds to the sequence of the order’s arrival.   
- result
![alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/FCFS_result.png)

#### B. Mixed Integer Programming(MIP)
- Algorithm
![alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/MIP.png)
The objective function in Equation (3) minimizes the total distance visited by all batches.   
Equation (4) ensures that the number of orders assigned to each batch does not exceed the maximum order count per batch i.e. number of bins.    
Equation (5) ensures that every order is assigned to one batch only. 
Equation (6) states that an aisle is assigned to a batch if at least one order in the batch requires that aisle.    
Equation (7) finds the last aisle visited by each batch,
then if Ymk = 1, therefore the aisle index is multiplied by Ymk.    
Equation (8) provides the upper bound for the last
aisle LastAk.   
Equation (9) counts the number of aisles assigned to each batch.   
Equation (10) calculates the estimated total traveled distance for a batch based on Equation (1).    
Finally, Equation (11) states the binary constraints for
𝐴𝐴𝑚𝑚𝑚𝑚, 𝑋𝑋𝑖𝑖𝑖𝑖, 𝑌𝑌𝑚𝑚𝑚𝑚, and Equatio (12) limits 𝑁𝑁𝑘𝑘 and 𝐷𝐷𝑘𝑘 to positive values only.   

- result
![alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/MIP_result.png)

#### C. Order Batching Heuristic
![alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/Heurisric.png)

![alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/heuristic_result.png)

### 3. Results

![alt text](/Study/Python/Machine_Learning/Paper/order_batching_optimization/imgs/Results_methods.png)
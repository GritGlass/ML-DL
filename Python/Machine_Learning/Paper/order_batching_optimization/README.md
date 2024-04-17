Order Batching Optimization for Warehouses with Clusrer-Picking   
Aaya Aboelfotoh etc.
===================================================================

## 25th International Conference on Production Research Manufacuring Innovation: Cyber Physical Manufacturing
## August 9-14, 2019, Chicago, Illinois(USA)

### 1. Problem
![problem_img](https://github.com/Kookkool/Study/assets/105410621/813ab486-1e95-4e1b-a917-23f2a4825590)       


### 2. Method
#### A. First Come First Serve(FCFS)
the order number also corresponds to the sequence of the order’s arrival.   
- result
![FCFS_result](https://github.com/Kookkool/Study/assets/105410621/2b0c14d0-aa9a-4175-9bdd-47b8d28ea870) 

#### B. Mixed Integer Programming(MIP)
- Algorithm  
![MIP](https://github.com/Kookkool/Study/assets/105410621/54aa3e5f-9a24-496f-b137-0f7ef6584a99)    
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
![MIP_result](https://github.com/Kookkool/Study/assets/105410621/1181caa9-41ea-473f-89be-7a5cc0446f92)    
    

#### C. Order Batching Heuristic   
![Heurisric](https://github.com/Kookkool/Study/assets/105410621/8460cbc2-f59a-4638-bbd9-96e801ed9a0f)       

![heuristic_result](https://github.com/Kookkool/Study/assets/105410621/fbc769bd-1fcd-4c3d-ab0c-8e0cbf1d058e)      

### 3. Results   

![Results_methods](https://github.com/Kookkool/Study/assets/105410621/3e9928ac-7d7e-484d-8fe3-9c686b4c0f0f)    
 
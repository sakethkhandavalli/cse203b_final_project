from scipy import optimize
import numpy as np
import queue
from math import ceil,floor
import copy
from static_features import get_static_features
from dynamic_features import get_dynamic_features
import os
import warnings
warnings.filterwarnings("ignore")
import time

# Sample MIP problem
c = [1,2,6,10,10,-20,30,-15,25,10]*3
A = [[5,0,7,2,-5,4,2,7,1,10,5,0,7,2,-5,4,2,7,1,10,5,0,7,2,-5,4,2,7,1,10],[0,0,4,5,-12,-15,20,30,5,6,0,0,4,5,-12,-15,20,30,5,6,0,0,4,5,-12,-15,20,30,5,6],[4,0,0,2,5,2,4,-1,-3,-10,4,0,0,2,5,2,4,-1,-3,-10,4,0,0,2,5,2,4,-1,-3,-10],[-1,-2,7,8,12,3,8,-2,-8,-5,-1,-2,7,8,12,3,8,-2,-8,-5,-1,-2,7,8,12,3,8,-2,-8,-5]]
b = [30,30,30,30]
bounds = [(0, None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None)]*3
int_const = [1,1,1,1,1,1,1,1,1,1]*3
Aeq = None
beq = None

def is_int(i):
	thres = 0.01
	return abs(i-round(i,1))<thres and float(round(i,1)).is_integer()

def solve_LP(bounds):
	# For min
	res = optimize.linprog(c, A, b, Aeq, beq,bounds=bounds,options={"disp": False})#,method=method)
	
	if res.success:
		return [res.x,sum([x*y for x,y in zip(c, res.x)])]
	else:
		return ['infeasible']

''' Need to optimize '''
def get_bound_A_b(bounds):
	#print('\n\nIN GET_BOUND')
	# print('bounds : ',len(bounds))
	#print('A : ',A)
	#print('b : ',b)
	bound_A = A[:]
	bound_b = b[:]
	
	for i,bound in enumerate(bounds):
		#print(bound[0],bound[1])
		if (bound[0] != None) and (bound[0]):
			# Greater than constraint
			row_g = np.zeros(len(bounds))
			#print("g")
			row_g[i] = -1
			#print(bound_A,row_g)
			bound_A.append(row_g)
			#print(bound_A)
			bound_b.append(bound[0])
		if (bound[1] != None) and (bound[1]):
			# Less than constraint
			row_l = np.zeros(len(bounds))
			#print('l')
			row_l[i] = 1
			bound_A.append(row_l)
			bound_b.append(bound[1])
	
	#print('bnd_A : ',bound_A)
	#print('bnd_b : ',bound_b)
	return bound_A,bound_b

labels = dict()
static_features = dict()
dynamic_features = dict()
candidates_at_node = dict()
infeasible_stats = dict()
features = dict()

num_costs = 0
pseudocost_up = 0
pseudocost_down = 0

def normalize_features(static_features,dynamic_features):
	for node in dynamic_features.keys():
		features[node] = dict()
		for var in dynamic_features[node]:
			features[node][var] = dynamic_features[node][var]
			features[node][var].extend(static_features[var])
		#print(features[node])
		all_f = np.array(list(features[node].values()))
		#print(all_f)
		a = (all_f - all_f.min(axis=0))
		b = (all_f.max(axis=0) - all_f.min(axis=0))
		for i,x in enumerate(b):
			if x==0:
				a[:,i] = np.ones(np.shape(a[:,i]))
				b[i] = 1
		all_f_normed = (a/b) + 1
		#print(all_f_normed)
		i=0 
		for key in features[node].keys():
			features[node][key] = all_f_normed[i,:]
			i+=1
	return features

def get_candidates(x):
	candidates = [i for i in range(len(int_const)) if (int_const[i]==1) and (not float(x[i]).is_integer())]
	return candidates

def get_children(x,cand,bounds):
	bounds1 = bounds[:]
	bounds2 = bounds[:]
	# Branch upward
	bounds1[cand] = (ceil(x[cand]),bounds[cand][1])		
	#print('bounds1,x[cand] : ',bounds1,x[cand])	
	res1 = solve_LP(bounds1)
	if res1[0] == 'infeasible':
		val1 = 10e8
	else:
		val1 = res1[1]

	#print('x : ',res1[0])
	# Branch downward
	bounds2[cand] = (bounds[cand][0],floor(x[cand]))
	#print('bounds2,x[cand] : ',bounds2,x[cand])
	res2 = solve_LP(bounds2)
	if res2[0] == 'infeasible':
		val2 = 10e8
	else:
		val2 = res2[1]
	#print('x : ',res2[0])

	return res1,res2,bounds1,bounds2,val1,val2

def strong_branch(x,val,bounds,node_no,static_features):
	global pseudocost_up
	global pseudocost_down
	global num_costs
	
	# print('Strong Branching on : ',x,val,bounds)
	# print('\n'*3)
	
	epsilon = -10e-6
	candidates = get_candidates(x)
	# print('candidates : ',candidates)

	candidates_at_node[node_no] = candidates

	select_score = -10e8
	select_var = ''
	select_bounds1 = []
	select_bounds2 = []
	select_x1 = [x,[]]
	select_x2 = [x,[]]
	sb_scores = dict()
	select_up_val = 0
	select_down_val = 0

	dynamic_features[node_no] = dict()

	for cand in candidates:
		#print('Candidate : ',cand)
		res1,res2,bounds1,bounds2,val1,val2 = get_children(x,cand,bounds)

		sb_score = (max(val1 - val,epsilon) * max(val2 - val,epsilon))
		#print('sb_score : ',sb_score)
		sb_scores[cand] = sb_score
		
		# Calculate dynamic features
		dynamic_features[node_no][cand] = []
		# print(bounds)
		bound_A,bound_b = get_bound_A_b(bounds)
		dynamic_features[node_no][cand] = get_dynamic_features(dynamic_features[node_no][cand],cand,x,pseudocost_up,pseudocost_down,bound_A,bound_b,candidates,static_features)

		if sb_score > select_score:
			select_score = sb_score
			select_up_val = val1
			select_down_val = val2
			select_x1 = res1
			select_x2 = res2
			select_bounds1 = bounds1
			select_bounds2 = bounds2
			select_var = cand
		#print('\n'*2)

	# Get labels
	alpha = 0.2
	sb_max = max(sb_scores.items())[1]
	labels[node_no] = dict()
	for cand in candidates:
		if sb_scores[cand] >= (1-alpha)*sb_max:
			labels[node_no][cand] = 1
		else:
			labels[node_no][cand] = 0
	
	# Calculating pseudo-costs
	pseudocost_up = (pseudocost_up*num_costs + select_up_val)/(num_costs+1)
	pseudocost_down = (pseudocost_down*num_costs + select_down_val)/(num_costs+1)
	num_costs+=1

	return select_var,select_x1,select_x2,select_bounds1,select_bounds2

def start_svm_train(x,no_nodes,static_features):
	global features

	for var in range(len(x)):
		infeasible_stats[var][2] = infeasible_stats[var][0]/no_nodes
		infeasible_stats[var][3] = infeasible_stats[var][1]/no_nodes
		static_features[var].extend(infeasible_stats[var])
	
	features = normalize_features(static_features,dynamic_features)

	print_to_file(no_nodes)

	print('\n\nTraining started\n\n')
	# Train the svm using the features generated
	os.system("./svm_rank_learn -c 2 features.dat model")
	print('\n\nTraining completed')
	
	return

def print_to_file(no_nodes):
	f = open("features.dat", "w")
	for i in range(no_nodes-1):
		f.write("# query "+str(i+1) + "\n")
		for cand in candidates_at_node[i+1]:
			f.write(str(labels[i+1][cand]+1) + " ")
			
			f.write("qid:" + str(i+1) + " ")
			list_len = len(features[i+1][cand])
			for j,feature in enumerate(features[i+1][cand]):
				if j!=list_len-1:
					f.write(str(j+1) + ":" + str(feature) + " ")
				else:
					f.write(str(j+1) + ":" + str(feature)+"\n")

	f.close()
	return

def get_branch_var(x,static_features,bounds):
	candidates = get_candidates(x)
	f = open("test.dat", "w")
	dy_features = dict()
	dy_features[1] = dict()
	for cand in candidates:
		tmp_arr = []
		bound_A,bound_b = get_bound_A_b(bounds)		
		dy_features[1][cand] = get_dynamic_features(tmp_arr,cand,x,pseudocost_up,pseudocost_down,bound_A,bound_b,candidates,static_features)

	features = normalize_features(static_features,dy_features)
	
	for i,cand in enumerate(candidates):
		f.write(str(i+1) + " ")
		f.write("qid:" + str(1) + " ")
		list_len = len(features[1][cand])
		for j,feature in enumerate(features[1][cand]):
			if j!=list_len-1:
				f.write(str(j+1) + ":" + str(feature) + " ")
			else:
				f.write(str(j+1) + ":" + str(feature)+"\n")
	f.close()
	# Test File Created
	os.system("./svm_rank_classify test.dat model predictions > /dev/null")
	# Predictions found

	''' Now we iterate over predictions and select the variable with highest value'''
	f = open("predictions", "r")
	values = [float(line.strip()) for line in f]
	select_cand = values.index(max(values))
	#print('Selected Candidate : ',candidates[select_cand])

	''' Get the children LP after branching using the selected variable '''
	res1,res2,bounds1,bounds2,val1,val2 = get_children(x,candidates[select_cand],bounds)

	return select_cand,res1,res2,bounds1,bounds2

def ML_algorithm(problem, def_problem=True):
	global c, A, b, bounds, int_const, Aeq, beq
	if not def_problem:
		[c, A, b, bounds, int_const, Aeq, beq] = problem

	start_time = time.time()
	Q = queue.Queue()
	x = solve_LP(bounds)
	Q.put((bounds,x))
	opt_value = 10e8
	opt_x = []
	bounds1 = bounds[:]
	bounds2 = bounds[:]
	
	# Compute static features
	static_features = get_static_features(c,np.array(A))

	theta = 100
	no_nodes = 0
	sb_node = 0
	for var in range(len(x[0])):
		infeasible_stats[var] = [0,0,0,0]

	# Becomes 1 when svm is trained and is used for ranking
	svm_flag = 0

	while 1:
		if Q.qsize() == 0:
			break
		curr = Q.get()
		if curr[1][0] == 'infeasible':
			continue

		x = curr[1][0]
		obj_z = curr[1][1]
		bnds = curr[0]
		ip_fl=1
		for ind,i in enumerate(x):
			if (int_const[ind] == 1) and (is_int(i)):
				x[ind] = round(i)
			elif (int_const[ind] == 0):
				x[ind] = i
			else:
				ip_fl=0

		if obj_z > opt_value:
			# Skipped as LP obj. is greater than the best possible till now
			continue
		if ip_fl:
			# IP solution hence compare with best possible , no need to branch further
			if obj_z < opt_value:
				opt_value = obj_z
				opt_x = x
			continue
		# If a solution exists increment no_nodes
		no_nodes += 1

		if no_nodes > theta:
			if not svm_flag:
				svm_flag = 1
				start_svm_train(x,no_nodes,static_features)
			var,x1,x2,bounds1,bounds2 = get_branch_var(x,static_features,bnds)

		else:	
			var,x1,x2,bounds1,bounds2 = strong_branch(x,obj_z,curr[0],no_nodes,static_features)
			if x1[0] == 'infeasible' and x2[0] == 'infeasible':
				infeasible_stats[var][1]+=1
			elif x1[0] == 'infeasible' or x2[0] == 'infeasible':
				infeasible_stats[var][0]+=1
		
		#print('Selected variable and its subproblems : x'+str(var))
		#print('x1,x2 : ',x1,x2)

		#print('bounds1 : ',bounds1)
		#print('bounds2 : ',bounds2)

		Q.put((bounds1,x1))
		Q.put((bounds2,x2))
	
	end_time = time.time()
	total_time = end_time - start_time

	print("Optimal x : ",opt_x)
	print("Optimal value : ",sum([x*y for x,y in zip(c, opt_x)]))
	print('No_Nodes : ',no_nodes)

	return no_nodes, total_time

if __name__ == "__main__":
	ML_algorithm(None)

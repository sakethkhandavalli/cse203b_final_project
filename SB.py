from scipy import optimize
import numpy as np
import queue
from math import ceil,floor
import warnings
warnings.filterwarnings("ignore")
import time

# Sample MIP problem
c = [1,2,6,10,10,-20,30,-15,25,10]
A = [[5,0,7,2,-5,4,2,7,1,10],[0,0,4,5,-12,-15,20,30,5,6],[4,0,0,2,5,2,4,-1,-3,-10],[-1,-2,7,8,12,3,8,-2,-8,-5]]
b = [30,30,30,30]
bounds = [(0, None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None)]
int_const = [1,1,1,1,1,1,1,1,1,1]
Aeq = None
beq = None

def is_int(i):
	thres = 0.01
	return abs(i-round(i,1))<thres and float(round(i,1)).is_integer()

def solve_LP(bounds):
	# For minimization problem
	res = optimize.linprog(c, A, b, Aeq, beq,bounds=bounds,options={"disp": False})#,method=method)
	
	if res.success:
		return [res.x,sum([x*y for x,y in zip(c, res.x)])]
	else:
		return ['infeasible']

labels = dict()
candidates_at_node = dict()
features = dict()

def get_candidates(x):
	candidates = [i for i in range(len(int_const)) if (int_const[i]==1) and (not float(x[i]).is_integer())]
	return candidates

def get_children(x,cand,bounds):
	bounds1 = bounds[:]
	bounds2 = bounds[:]
	# Branch upward
	bounds1[cand] = (ceil(x[cand]),bounds[cand][1])		
	res1 = solve_LP(bounds1)
	if res1[0] == 'infeasible':
		val1 = 10e8
	else:
		val1 = res1[1]

	# Branch downward
	bounds2[cand] = (bounds[cand][0],floor(x[cand]))
	res2 = solve_LP(bounds2)
	if res2[0] == 'infeasible':
		val2 = 10e8
	else:
		val2 = res2[1]

	return res1,res2,bounds1,bounds2,val1,val2

def strong_branch(x,val,bounds):
	epsilon = -10e-6
	candidates = get_candidates(x)

	select_score = -10e8
	select_bounds1 = []
	select_bounds2 = []
	select_x1 = [x,[]]
	select_x2 = [x,[]]

	for cand in candidates:
		res1,res2,bounds1,bounds2,val1,val2 = get_children(x,cand,bounds)

		sb_score = (max(val1 - val,epsilon) * max(val2 - val,epsilon))
		
		if sb_score > select_score:
			select_score = sb_score
			select_x1 = res1
			select_x2 = res2
			select_bounds1 = bounds1
			select_bounds2 = bounds2

	return select_x1,select_x2,select_bounds1,select_bounds2

def SB(problem, def_problem=True):
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
	
	no_nodes = 0

	while 1:
		if Q.qsize() == 0:
			break
		curr = Q.get()
		if curr[1][0] == 'infeasible':
			continue

		x = curr[1][0]
		obj_z = curr[1][1]
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
		x1,x2,bounds1,bounds2 = strong_branch(x,obj_z,curr[0])
		
		Q.put((bounds1,x1))
		Q.put((bounds2,x2))
	
	end_time = time.time()
	total_time = end_time - start_time
	print("Optimal x : ",opt_x)
	print("Optimal value : ",sum([x*y for x,y in zip(c, opt_x)]))
	print('No_Nodes : ',no_nodes)

	return no_nodes, total_time

if __name__ == "__main__":
	SB(None)

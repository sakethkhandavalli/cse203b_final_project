import numpy as np
from math import ceil,floor

def get_dynamic_features(dynamic_features,cand,full_x,up_val,down_val,A,b,candidates,static_features):
	x = full_x[cand]
	A = np.array(A)
	# print('In dy_feat x : ',x)
	# print('In dy_feat A : ',A)
	# print('In dy_feat b : ',b)
	# print(static_features)
	# Slack and ceil distances
	dynamic_features.append(min(x-floor(x),ceil(x)-x))
	dynamic_features.append(ceil(x)-x)

	# Pseudocosts
	fraction = x-floor(x)
	# print('up,down : ',up_val,down_val)
	dynamic_features.append(up_val)
	dynamic_features.append(down_val)
	dynamic_features.append((up_val*(1-fraction)) + (fraction*down_val))
	dynamic_features.append((up_val*(1-fraction)) * (fraction*down_val))
	dv = down_val
	if down_val == 0:
		dv=0.00001
	dynamic_features.append((up_val*(1-fraction)) / (fraction*dv))

	# Stats. for constraint degrees
	ind = np.nonzero(A[:,cand])[0]
	part_cons = A[ind,:]
	cons_degree = []
	for cons in part_cons:
		cons_degree.append(len(np.nonzero(cons)[0]))
	mean = np.mean(cons_degree)
	min_v = np.min(cons_degree)
	max_v = np.max(cons_degree)
	dynamic_features.extend([mean,np.std(cons_degree)])
	dynamic_features.extend([min_v,max_v])
	dynamic_features.extend([static_features[cand][5]/mean , static_features[cand][7]/min_v, static_features[cand][8]/max_v])

	# Min/max for ratios of constraint coeffs. to RHS
	p_rhs = list(filter(lambda r: r[1]>0, enumerate(b)))
	n_rhs = list(filter(lambda r: r[1]<0, enumerate(b)))
	ind_p = [p[0] for p in p_rhs]
	val_p = [p[1] for p in p_rhs]
	ind_n = [n[0] for n in n_rhs]
	val_n = [n[1] for n in n_rhs]
	
	x_coeff = A[:,cand][ind_p]
	ratio_p = [x_c/p for x_c,p in zip(x_coeff,val_p)]
	if len(ratio_p)==0:
		ratio_p = [0]
	dynamic_features.extend([np.min(ratio_p),np.max(ratio_p)])

	x_coeff = A[:,cand][ind_n]
	ratio_n = [x_c/n for x_c,n in zip(x_coeff,val_n)]
	if len(ratio_n)==0:
		ratio_n = [0]
	dynamic_features.extend([np.min(ratio_n),np.max(ratio_n)])

	# Min/Max for one-to-all coefficient ratios
	A = np.array(A)

	p_to_p = []
	p_to_n = []
	n_to_p = []
	n_to_n = []
	for cons in A:
		positive_coeffs = list(filter(lambda r: r>0, cons))
		negative_coeffs = list(filter(lambda r: r<0, cons))
		sum_p = 0.00001 if sum(positive_coeffs)==0 else sum(positive_coeffs)
		sum_n = 0.00001 if sum(negative_coeffs)==0 else sum(negative_coeffs)
		neg_x = 0 if x>0 else x
		pos_x = 0 if x<0 else x
		p_to_p.append(pos_x/sum_p)
		p_to_n.append(pos_x/sum_n)
		n_to_p.append(neg_x/sum_p)
		n_to_n.append(neg_x/sum_n)
	dynamic_features.extend([min(p_to_p),max(p_to_p),min(p_to_n),max(p_to_n)])
	dynamic_features.extend([min(n_to_p),max(n_to_p),min(n_to_n),max(n_to_n)])

	# Active constraint coeffs
	active_ind = []
	for i,cons in enumerate(A):
		if sum([a*b for a,b in zip(cons, full_x)]) == b[i]:
			active_ind.append(i)

	x_coeff = A[:,cand][active_ind]
	# print(x_coeff)
	w_schemes = [np.ones(len(active_ind))]
	arr1 = []
	arr2 = []
	for cons in A[active_ind,:]:
		cons = np.array(cons)
		sum_1 = sum(cons)
		sum_2 = sum(cons[candidates])
		if sum_1:
			arr1.append(1/sum_1)
		else:
			arr1.append(1000000)
		if sum_2:
			arr2.append(1/sum_2)
		else:
			arr2.append(1000000)

	w_schemes.append(arr1)
	w_schemes.append(arr2)

	x_part = np.array(x_coeff)
	ind = np.nonzero(x_part)[0]
	x_part[ind] = 1

	for scheme in w_schemes:
		weighted_x = np.multiply(scheme,x_coeff)
		if len(weighted_x)==0:
			weighted_x = [0]
		dynamic_features.extend([np.sum(weighted_x),np.mean(weighted_x),np.std(weighted_x)])
		dynamic_features.extend([np.min(weighted_x),np.max(weighted_x)])
		weighted_part = np.dot(scheme,x_part)
		dynamic_features.append(weighted_part)

	return dynamic_features
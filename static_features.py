import numpy as np

static_features = dict()

def get_static_features(obj_fun,constraints):
	no_variables = len(obj_fun)

	for var in range(no_variables):
		# Objective function coeffs.
		static_features[var] = [obj_fun[var]]
		if obj_fun[var]>=0:
			static_features[var].append(obj_fun[var])
			static_features[var].append(0)
		else:
			static_features[var].append(0)
			static_features[var].append(obj_fun[var])

		# No. of constraints
		# Constraints in which x_var participates
		ind = np.nonzero(constraints[:,var])[0]
		static_features[var].append(len(ind))

		# Stats for constraint degrees
		part_cons = constraints[ind,:]
		cons_degree = []
		for cons in part_cons:
			cons_degree.append(len(np.nonzero(cons)[0]))
		
		static_features[var].extend([np.mean(cons_degree),np.std(cons_degree)])
		static_features[var].extend([np.min(cons_degree),np.max(cons_degree)])

		# Stats for constraint coeffs.
		positive_coeffs = list(filter(lambda x: x>0, part_cons[:,var]))
		negative_coeffs = list(filter(lambda x: x<0, part_cons[:,var]))

		if len(positive_coeffs) > 0:
			static_features[var].extend([len(positive_coeffs),np.mean(positive_coeffs),np.std(positive_coeffs)])
			static_features[var].extend([np.min(positive_coeffs),np.max(positive_coeffs)])
		else:
			static_features[var].extend([0,0,0,0,0])

		if len(negative_coeffs) > 0:
			static_features[var].extend([len(negative_coeffs),np.mean(negative_coeffs),np.std(negative_coeffs)])
			static_features[var].extend([np.min(negative_coeffs),np.max(negative_coeffs)])		
		else:
			static_features[var].extend([0,0,0,0,0])

	return static_features
import numpy as np


# simple rref algorithm
def rref(m):
	matrix = np.copy(m)
	for rc in range(len(matrix)):
		lead = False

		for ind in range(rc, len(matrix)):
			if not(lead) and matrix[ind,rc] != 0:

				lead = True
				matrix[ind] = matrix[ind] / matrix[ind,rc]

				temp = np.copy(matrix[rc])

				matrix[rc] = matrix[ind]

				matrix[ind] = temp
		if (lead):
			for ind in range(len(matrix)):
				if ind != rc:
					matrix[ind] = matrix[ind] - (matrix[ind,rc]*matrix[rc])

	return matrix

# linear system solution function (sets free variables to 0)
def solve(A, Y):
	aug = np.append(A, Y.reshape((Y.shape[0], 1)), axis=1)

	clean = rref(aug)

	coeffs = np.zeros(len(aug[0])-1)

	for row in clean:
		leadingOne = -1
		ind = 0
		if (row[-1] != 0):
			for el in row[:-1]:
				if (el == 1.0 and leadingOne == -1):
					leadingOne = ind
					coeffs[ind] = row[-1]
				elif (el != 0.0):
					coeffs[ind] = 0
				ind += 1

	return coeffs

def twoDpolyEval(coeffs, x, y):
	z = 0

	xPow = 1
	for row in coeffs:
		yPow = 1
		for coeff in row:
			z += coeff * xPow * yPow
			yPow *= y
		xPow *= x

	return z
def sigma(ps, cs, zDeg=0):
	sum = 0
	for point in ps:
		for pos, coord in zip(point[:-1], cs):
			sum += pos**coord
		sum += point[-1]**zDeg
	return sum

def sigma(ps, xDeg, yDeg, zDeg=0):
	sum = 0
	for x, y, z in ps:
		sum += (x**xDeg)*(y**yDeg)*(z**zDeg)
	return sum

def twoDpolyFit(ps, xDeg, yDeg):
	A = np.zeros(((xDeg+1)*(yDeg+1), (xDeg+1)*(yDeg+1)))
	ps = np.array(ps, dtype="float")

	for r in range((xDeg+1)*(yDeg+1)):
		xRow = r // (yDeg+1)
		yRow = r % (yDeg+1)
		for c in range((xDeg+1)*(yDeg+1)):
			xCol = c // (yDeg+1)
			yCol = c % (yDeg+1)

			A[r, c] = sigma(ps, xRow+xCol, yRow+yCol)

	Z = np.zeros((xDeg+1)*(yDeg+1))

	for t in range((xDeg+1)*(yDeg+1)):
		xTow = t // (yDeg+1)
		yTow = t % (yDeg+1)

		Z[t] = sigma(ps, xTow, yTow, zDeg=1)

	cS = solve(A, Z)
	coeffs = cS.reshape((xDeg+1),(yDeg+1))
	return coeffs

def nDpolyFit(ps, *degs):
	ps = np.array(ps, dtype="float")
	degComb = 1
	for deg in degs:
		degComb *= deg + 1

	A = np.zeros((degComb, degComb))

	for r in range(degComb):
		num = r
		coords = np.zeros(len(degs))
		degSoFar = 1
		for ind in range(len(degs)-1, -1, -1):
			deg = degs[ind]
			coords[ind] += (num // degSoFar) % (deg+1)
			
			degSoFar *= (deg+1)

  
		for c in range(degComb):
			num = c
			degSoFar = 1
			cs = coords.copy()
			for ind in range(len(degs)-1, -1, -1):
				deg = degs[ind]
				cs[ind] += (num // degSoFar) % (deg+1)
				
				degSoFar *= (deg+1)


			A[r, c] = sigma(ps, cs)

	Z = np.zeros(degComb)
 
	for t in range(degComb):
		num = t
		coords = np.zeros(len(degs))
		degSoFar = 1
		for ind in range(len(degs)-1, -1, -1):
			deg = degs[ind]
			coords[ind] += (num // degSoFar) % (deg+1)
			
			degSoFar *= (deg+1)

		Z[t] = sigma(ps, coords, zDeg=1)
  
	cS = solve(A, Z)
	outShape = np.array(degs) + 1
	coeffs = cS.reshape(outShape)
	return coeffs
	
			
		

def nDpolyEval(coeffs, *pos):
	z = 0

	return z
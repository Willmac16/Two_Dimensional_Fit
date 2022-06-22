from re import T
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

def sigma(ps, cs, zDeg=0):
	sum = 0
	for point in ps:
		s = 1
		for pos, coord in zip(point[:-1], cs):
			s *= pos**coord
		sum += s * point[-1]**zDeg
	return sum

def legacy_sigma(ps, xDeg, yDeg, zDeg=0):
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

			A[r, c] = legacy_sigma(ps, xRow+xCol, yRow+yCol)
 
	Z = np.zeros((xDeg+1)*(yDeg+1))

	for t in range((xDeg+1)*(yDeg+1)):
		xTow = t // (yDeg+1)
		yTow = t % (yDeg+1)

		Z[t] = legacy_sigma(ps, xTow, yTow, zDeg=1)

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
		coords = np.zeros(len(degs), dtype="int")
		degSoFar = 1
		for ind in range(len(degs)-1, -1, -1):
			deg = degs[ind]
			coords[ind] += (r // degSoFar) % (deg+1)
			
			degSoFar *= (deg+1)

  
		for c in range(degComb):
			degSoFar = 1
			cs = coords.copy()
			for ind in range(len(degs)-1, -1, -1):
				deg = degs[ind]
				cs[ind] += (c // degSoFar) % (deg+1)
				
				degSoFar *= (deg+1)
    
			A[r, c] = sigma(ps, cs)
 
	Z = np.zeros(degComb)
 
	for t in range(degComb):
		coords = np.zeros(len(degs), dtype="int")
		degSoFar = 1
		for ind in range(len(degs)-1, -1, -1):
			deg = degs[ind]
			coords[ind] += (t // degSoFar) % (deg+1)
			
			degSoFar *= (deg+1)

		Z[t] = sigma(ps, coords, zDeg=1)
  
	cS = solve(A, Z)
	outShape = np.array(degs) + 1
	coeffs = cS.reshape(outShape)
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

def nDpolyEval(coeffs, *pos):
	z = 0
	pos = np.array(pos)

	degComb = 1
	for deg in coeffs.shape:
		degComb *= deg

	for t in range(degComb):
		coords = np.zeros(len(coeffs.shape), dtype="int")
		degSoFar = 1
		for ind in range(len(coeffs.shape)-1, -1, -1):
			ord = coeffs.shape[ind]
			coords[ind] += (t // degSoFar) % ord
			
			degSoFar *= ord
		z += coeffs[tuple(coords.tolist())] * np.product(np.power(pos, coords))

	return z

def test():
	print("Comparing 2D polynomial fitting to nD polynomial fitting")

	ps = [[0, 0, 4], [1, 0, -1], [2, 0, 0], [0, 2, 0],  [0, 1, -1], [1, 1, 1], [1, 2, 3], [2, 1, 3], [2, 2, 4]]

	twoD = twoDpolyFit(ps, 2, 2)
	nD = nDpolyFit(ps, 2, 2)
	print("Coeffs Match:", np.allclose(twoD, nD))

	fail = False
	for x, y, z in ps:
		fail = fail or twoDpolyEval(twoD, x, y) != nDpolyEval(nD, x, y)
	
	print("Eval Match:", not fail)
    
    
if __name__ == "__main__":
    test()
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
                    matrix[ind] -= (matrix[ind,rc]*matrix[rc])

    return matrix

# invert a matrix with rref
def invert(m):
    shape = m.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Matrix must be square")

    aug = np.zeros((shape[0], 2*shape[1]))

    aug[:,:shape[1]] = m
    aug[:,shape[1]:] = np.identity(shape[0])

    rrefed = rref(aug)

    out = rrefed[:, shape[1]:].copy()
    return out


# linear system solution function (sets free variables to 0)
def solve(A, Y):
    aug = np.append(A, Y.reshape((Y.shape[0], 1)), axis=1)

    # print(aug)
    clean = rref(aug)
    # print(clean)

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

def legacy_sigma(ps, xDeg: int, yDeg: int, zDeg=0):
    sum = 0
    for x, y, z in ps:
        sum += (x**xDeg)*(y**yDeg)*(z**zDeg)
    return sum

def twoDpolyFit(ps, xDeg: int, yDeg: int):
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

def nDpolyFit(ps, *degs: int):
    ps = np.array(ps, dtype="float")
    degComb = 1
    for deg in degs:
        degComb *= deg + 1

    A = np.zeros((degComb, degComb))

    # For each row
    for r in range(degComb):
        # Create a coordinate vector in polynomial dimension space
        coords = np.zeros(len(degs), dtype="int")
        degSoFar = 1

        # For each dimension, starting at the last and going back to the first compute the coords
        for ind in range(len(degs)-1, -1, -1):
            deg = degs[ind]
            coords[ind] += (r // degSoFar) % (deg+1)

            degSoFar *= (deg+1)

        # For each column do the same
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

        Z[t] = sigma(ps, coords, 1)

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

def nDpolyEval(coeffs: np.ndarray, *pos: float):
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

def oneDderivative(coeffs: np.ndarray):
    output_coeffs = np.zeros(len(coeffs) - 1)

    for ind, coeff in enumerate(coeffs[1:]):
        output_coeffs[ind-1] = ind * coeff

    return output_coeffs

def binary_search(x, points, monotonic_positive):
    ind_u = len(points) - 1
    ind_l = 0

    while ind_u != ind_l:
        ind_n = (ind_u - ind_l) // 2

        if (points[ind_u] - x) * (points[ind_l]) < 0:
            ind_u = ind_n
        else:
            ind_l = ind_n

    return ind_l

# Wrapper for polyeval that handles a 1D piecewise function if splits are given
class PiecewisePolyfit:
    def __init__(self, x, y, degree: int, splits):
        # Join the x and y arrays into a single array
        func = np.column_stack((x, y))
        func.sort(axis=1)

        # X needs to be monotonic
        self.monotonic_positive = True

        if x[-1] < x[0]:
            self.monotonic_positive = False

        ind = 0
        last_ind = 0
        # Split the x and y arrays at the split points
        arrays = []

        self.coeffs = []
        self.split_points = []

        splits.sort()

        # import matplotlib.pyplot as plt

        for split in splits:
            self.split_points.append(func[split - 1][0])

            arrays.append(func[last_ind:split])

            last_ind = split


        for array in arrays:
            self.coeffs.append(nDpolyFit(array, degree))

        self.splits = splits

    def eval(self, x: float):
        # Find the segment the x value is in
        ind = 0

        bin_ind = binary_search(x, self.split_points, monotonic_positive=self.monotonic_positive)
        if self.monotonic_positive:
            while ind + 1 < len(self.split_points) and x > self.split_points[ind]:
                ind += 1
        else:
            while ind + 1 < len(self.split_points) and x < self.split_points[ind]:
                ind += 1

        print("Binary Search Check: {}".format("passed" if bin_ind == ind else "failed"))

        segment = ind

        # print("seg {} num segs {}".format(segment, len(self.coeffs)))

        # Run polyeval
        y = nDpolyEval(self.coeffs[segment], np.array([x]))
        return y


def test():
    import cppimport.import_hook
    import cpp_tdf as cpp


    print("RREF Check")

    A = np.array([[2, 0, 2], [1, 2, 3], [1, 1, 1]])

    cpp.print_array(A)

    print(A)
    print(rref(A))
    print(cpp.rref(A))

    print("Comparing 2D polynomial fitting to nD polynomial fitting")

    ps = [[0, 0, 4], [1, 0, -1], [2, 0, 0], [0, 2, 0],  [0, 1, -1], [1, 1, 1], [1, 2, 3], [2, 1, 3], [2, 2, 4]]

    twoD = twoDpolyFit(ps, 2, 2)
    nD = nDpolyFit(ps, 2, 2)
    cpp_twoD = cpp.twoDpolyFit(ps, 2, 2)

    print("2D: {}".format(twoD))
    print("nD: {}".format(nD))
    print("cpp: {}".format(cpp_twoD))


    print("Coeffs Match:", np.allclose(twoD, nD))
    print("Coeffs Match:", np.allclose(twoD, cpp_twoD))


    print("Comparing cpp 3D polynomial fitting to nD polynomial fitting")

    ps_three = [[0, 0, 1, 4], [1, 0, 3, -1], [2, 0, 9, 0], [0, 2, 2, 0],  [0, 1, 7, -1], [1, 1, -2, 1], [1, 2, 9, 3], [2, 1, 11, 3], [2, 2, -3, 4]]

    nD_three = nDpolyFit(ps_three, 2, 2, 2)
    cpp_threeD = cpp.threeDpolyFit(ps_three, 2, 2, 2)

    print("Coeffs Match:", np.allclose(nD_three, cpp_threeD))


    # fail = False
    # fail_two = False
    # for x, y, z in ps:
    #     fail = fail or twoDpolyEval(twoD, x, y) != nDpolyEval(nD, x, y)
    #     # fail_two = fail_two or twoDpolyEval(twoD, x, y) != cpp.twoDpolyEval(cpp_twoD, x, y)


    # print("Eval Match:", not fail)
    # # print("Eval Match:", not fail)


    # print("Testing 1d case")
    # ps = [[10, 10], [11, -50], [19, 55], [20, -5], [30, 0]]
    # oneD = nDpolyFit(ps, 4)
    # import matplotlib.pyplot as plt

    # xs = np.linspace(0, 30, 100)

    # plt.plot([point[0] for point in ps], [point[1] for point in ps], label="Data")
    # plt.plot(xs, [nDpolyEval(oneD, x) for x in xs], label="Polynomial")

    # plt.legend()
    # plt.show()

    # print("Invert Test")

    # rot = np.array(((0, -1), (1, 0)))
    # print(rot)
    # print(invert(rot))
    # print(rot @ invert(rot))

if __name__ == "__main__":
    test()
    # ppTest()

import numpy as np



a = np.array ([[1, 2, 3], [4, 5, 6]])

b = np.array ([1, 2, 3, 4, 5, 6])
b = b.reshape (3, 2)

c = np.array ([['a0', 'a1', 'a2', 'a3'],
               ['b0', 'b1', 'b2', 'b3'],
               ['c0', 'c1', 'c2', 'c3'],
               ['d0', 'd1', 'd2', 'd3'],
               ['e0', 'e1', 'e2', 'e3'],
               ['f0', 'f1', 'f2', 'f3'],
              ])


c = c.reshape (-1)


c_y = np.array (['a_y', 'b_y', 'c_y', 'd_y', 'e_y', 'f_y'])

d = np.array ([
               [['A0', 'A1', 'A2', 'A3'],
               ['B0', 'B1', 'B2', 'B3']],
               [['C0', 'C1', 'C2', 'C3'],
               ['D0', 'D1', 'D2', 'D3']],
               [['E0', 'E1', 'E2', 'E3'],
               ['F0', 'F1', 'F2', 'F3']],
              ])



print ('Shape:', a.shape)
print (a)

print ('Shape:', b.shape)
print (b)

print ('Shape:', c.shape)
print (c)

print ('Shape:', c_y.shape)
print (c_y)


print ('Shape:', d.shape)
print (d)


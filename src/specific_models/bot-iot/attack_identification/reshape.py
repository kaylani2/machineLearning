import numpy as np

STEPS = 2

def window_stack(a, stride=1, numberOfSteps=3):
    return np.hstack([ a [i:1+i-numberOfSteps or None:stride] for i in range(0,numberOfSteps) ])

#def window_stack(myArray, stride=1, numberOfSteps=3):
#    n = myArray.shape [0]
#    #for i in range (0, numberOfSteps):
#      #np.hstack (myArray [i: 1+n+i-numberOfSteps:stride]
#    return np.hstack( myArray [i:1+n+i-numberOfSteps:stride] for i in range(0,numberOfSteps) )


a = np.array ( [[1, 2, 3], [4, 5, 6]])

b = np.array ( [1, 2, 3, 4, 5, 6])
b = b.reshape (3, 2)

c = np.array ( [['a0', 'a1', 'a2', 'a3'],
               ['b0', 'b1', 'b2', 'b3'],
               ['c0', 'c1', 'c2', 'c3'],
               ['d0', 'd1', 'd2', 'd3'],
               ['e0', 'e1', 'e2', 'e3'],
               ['f0', 'f1', 'f2', 'f3'],
               ['f0', 'f1', 'f2', 'f3'],
              ])


#c = c.reshape (-1)
#if ( (c.shape [0] % STEPS) != 0):
#  #c = np.delete (c, c.shape [0] % STEPS)
#  c = c [:-(c.shape [0] % STEPS), :]
#
#c = c.reshape ((c.shape [0] // STEPS, STEPS, 4), order = 'C')



c_y = np.array ( ['a_y',
                 'b_y',
                 'c_y',
                 'd_y',
                 'e_y',
                 'f_y',
                 'f_y',
                ])
#if ( (c_y.shape [0] % STEPS) != 0):
  #c = np.delete (c, c.shape [0] % STEPS)
  #c_y = c_y [:-(c.shape [0] % STEPS)]
#c_y = c_y.reshape ((3, STEPS), order = 'C')

d = np.array ( [
               [ ['A0', 'A1', 'A2', 'A3'],
               ['B0', 'B1', 'B2', 'B3']],
               [ ['C0', 'C1', 'C2', 'C3'],
               ['D0', 'D1', 'D2', 'D3']],
               [ ['E0', 'E1', 'E2', 'E3'],
               ['F0', 'F1', 'F2', 'F3']],
              ])



print ('Shape:', a.shape)
print (a)

print ('Shape:', b.shape)
print (b)

#c_y = c_y [4:]
print ('Shape:', c_y.shape)
print (c_y )


print ('Shape:', c.shape)
print (c)

print ('Shape:', d.shape)
print (d)


c = np.array ( [['a0', 'a1', 'a2', 'a3'],
               ['b0', 'b1', 'b2', 'b3'],
               ['c0', 'c1', 'c2', 'c3'],
               ['d0', 'd1', 'd2', 'd3'],
               ['e0', 'e1', 'e2', 'e3'],
               ['f0', 'f1', 'f2', 'f3'],
               ['f0', 'f1', 'f2', 'f3'],
              ])

c = np.arange(2201112 * 9).reshape(2201112, 9)

#c = np.zeros((2201112, 9))


#print ('Shape:', c.shape)
#print (c)
#
#c = window_stack (c, 1, 2)
#print ('Shape:', c.shape)
#print (c)
#
#c = c.reshape (2201111, 2, 9)
#print ('Shape:', c.shape)
#print (c)


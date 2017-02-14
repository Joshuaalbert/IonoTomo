
# coding: utf-8

# In[44]:

from sympy import symbols, Function, Rational,Matrix,cse

def zparam():
    px,py,pz,x,y,z,s = symbols('px py pz x y z s')
    n = Function('n')(x,y,z)
    nx = n.diff(x)
    ny = n.diff(y)
    nz = n.diff(z)
    sdot = n/pz
    pxdot = nx*n/pz
    pydot = ny*n/pz
    pzdot = nz*n/pz
    xdot = px/pz
    ydot = py/pz
    zdot = Rational(1)
    euVec = Matrix([pxdot,pydot,pzdot,xdot,ydot,zdot,sdot]).T
    jac = Matrix([euVec.diff(px)[:],
         euVec.diff(py)[:],
         euVec.diff(pz)[:],
         euVec.diff(x)[:],
         euVec.diff(y)[:],
         euVec.diff(z)[:],
         euVec.diff(s)[:]])
    cseFunc = cse(jac.T,optimizations='basic')
    print(cseFunc)
    
def sparam():
    px,py,pz,x,y,z,s = symbols('px py pz x y z s')
    n = Function('n')(x,y,z)
    nx = n.diff(x)
    ny = n.diff(y)
    nz = n.diff(z)
    sdot = Rational(1)
    pxdot = nx
    pydot = ny
    pzdot = nz
    xdot = px/n
    ydot = py/n
    zdot = pz/n
    euVec = Matrix([pxdot,pydot,pzdot,xdot,ydot,zdot,sdot]).T
    jac = Matrix([euVec.diff(px)[:],
         euVec.diff(py)[:],
         euVec.diff(pz)[:],
         euVec.diff(x)[:],
         euVec.diff(y)[:],
         euVec.diff(z)[:],
         euVec.diff(s)[:]])
    cseFunc = cse(jac.T,optimizations='basic')
    print(cseFunc)
    
if __name__=="__main__":
    zparam()
    sparam()


# In[ ]:





# coding: utf-8

# In[17]:


    

def generateBinv():
    from sympy import symbols, Rational, Matrix
    from sympy.vector import Vector

    x,y,z = symbols('x y z')
    wholeFunc = Rational(0)
    alphaVec = []
    i = 0
    while i <= 3:
        j = 0
        while j <= 3:
            k = 0
            while k <= 3:
                a = symbols('a_{0}'.format(k+4*(j + 4*i)))
                alphaVec.append(a)
                wholeFunc += a*x**Rational(i)*y**Rational(j)*z**Rational(k)
                k += 1
            j += 1
        i += 1

    #print(wholeFunc)

    #order per corner f,fx,fy,fz,fxy,fxz,fyz,fxyz
    cornerVec = Matrix([wholeFunc,
                 wholeFunc.diff(x),
                 wholeFunc.diff(y),
                 wholeFunc.diff(z),
                 wholeFunc.diff(x,y),
                 wholeFunc.diff(x,z),
                 wholeFunc.diff(y,z),
                 wholeFunc.diff(x,y,z)])
    #print(cornerVec)
    #build B suchthat b = B.a

    def getBRows(vec,alpha):
        '''get the rows from cornervec evaluated at the specific points'''
        B = []
        for el in vec:
            row = []
            for a in alpha:
                k = Rational(el.diff(a))
                row.append(k)
            B.append(row)
        return Matrix(B)

    #x=0,y=0,z=0
    row0 = getBRows(cornerVec.subs({x:0,y:0,z:0}),alphaVec)
    row1 = getBRows(cornerVec.subs({x:0,y:0,z:1}),alphaVec)
    row2 = getBRows(cornerVec.subs({x:0,y:1,z:0}),alphaVec)
    row3 = getBRows(cornerVec.subs({x:0,y:1,z:1}),alphaVec)
    row4 = getBRows(cornerVec.subs({x:1,y:0,z:0}),alphaVec)
    row5 = getBRows(cornerVec.subs({x:1,y:0,z:1}),alphaVec)
    row6 = getBRows(cornerVec.subs({x:1,y:1,z:0}),alphaVec)
    row7 = getBRows(cornerVec.subs({x:1,y:1,z:1}),alphaVec)
    Binv = Matrix([row0,row1,row2,row3,row4,row5,row6,row7]).inv()
    string = "["
    for i in range(64):
        string += '['+('{:},'*63).format(*Binv[i,:64])+'{:}'.format(Binv[i,-1])+'],\n'
    string += ']'
    print(string)
    
def optimizeBvecFormation(order=1):
    from sympy import symbols, Matrix
    from sympy import cse, Function, Rational
    from scipy.special import binom
    
    def centralDiff(order, inFunc,var):
        #args = inFunc.args
        #func = inFunc.func
        #terms = []
        x,y,z,h = symbols('x y z h')
        outFunc = 0
        k = 0
        while k <= order:
            if var == 'x':
                outFunc += Rational(-1)**Rational(k) * Rational(int(binom(order,k))) * inFunc.subs({x:x + (Rational(order,2) - Rational(k))*h})
            if var == 'y':
                outFunc += Rational(-1)**Rational(k) * Rational(int(binom(order,k))) * inFunc.subs({y:y + (Rational(order,2) - Rational(k))*h})
            if var == 'z':
                outFunc += Rational(-1)**Rational(k) * Rational(int(binom(order,k))) * inFunc.subs({z:z + (Rational(order,2) - Rational(k))*h})
            k += 1
        return outFunc
    
    def centralDiff(order, inFunc,var):
        #args = inFunc.args
        #func = inFunc.func
        #terms = []
        x,y,z = symbols('xi yi zi', intergers=True)
        h = symbols('h')
        #print(var)
        if var == 'x':
            var = x
        if var == 'y':
            var = y
        if var == 'z':
            var = z
        outFunc = Rational(8)*(inFunc.subs({var:(var + h)}) - inFunc.subs({var:(var - h)}))/Rational(12)/h + (inFunc.subs({var:(var - Rational(2)*h)}) - inFunc.subs({var:(var + Rational(2)*h)}))/Rational(12)/h
        #print(inFunc)
        #print(Rational(8)*inFunc.subs({var:(var + h)})/Rational(12)/h)
        #print(-Rational(8)*inFunc.subs({var:(var - h)})/Rational(12)/h)
        #print(inFunc.subs({var:(var - Rational(2)*h)})/Rational(12)/h)
        #print(-inFunc.subs({var:(var + Rational(2)*h)})/Rational(12)/h)
        #outFunc = (inFunc.subs({var:var + h}) - inFunc.subs({var:var}))/h
        return outFunc
    
    vec = []
    
    xi,yi,zi,nz,ny = symbols('xi yi zi nz ny', intergers=True)
    func = Function('f')
    xf,yf,zf = Function('xf'),Function('yf'),Function('zf')
    vec = []
    #order
    #000,001,010,011,100,101,110,111
    i = 0
    while i <= 1:
        j = 0
        while j <= 1:
            k = 0
            while k <= 1:
                
                dx = (xf(xi+Rational(i+1)) - xf(xi+Rational(i-1)))/Rational(2)
                dy = (yf(yi+Rational(j+1)) - yf(yi+Rational(j-1)))/Rational(2)
                dz = (zf(zi+Rational(k+1)) - zf(zi+Rational(k-1)))/Rational(2)
                #f 
                f = func((zi+Rational(k)) + nz*((yi+Rational(j)) + ny*(xi+Rational(i))))#x+Rational(i),y+Rational(j),z+Rational(k))
                vec.append(f)
                
                #fx
                fx = centralDiff(order,f,'x')/dx
                vec.append(fx)
                
                #fy
                fy = centralDiff(order,f,'y')/dy
                vec.append(fy)
                
                #fz
                fz = centralDiff(order,f,'z')/dz
                vec.append(fz)
                
                #fxy
                fxy = centralDiff(order,fx,'y')/dy
                vec.append(fxy)
                
                #fxz
                fxz = centralDiff(order,fx,'z')/dz
                vec.append(fxz)
                
                #fyz
                fyz = centralDiff(order,fy,'z')/dz
                vec.append(fyz)
                
                #fxyz
                fxyz = centralDiff(order,fxy,'z')/dz
                vec.append(fxyz)
                
                k += 1
            j += 1
        i += 1
    vec = Matrix(vec).subs({'h':1})
    #print(vec)
    cseFunc = cse(vec,optimizations='basic')
    lines = []
    for optLine in cseFunc[0]:
        line = "{0} = {1}".format(optLine[0],optLine[1].evalf())
        line = line.replace("xf","self.get_xvec").replace("yf","self.get_yvec").replace("zf","self.get_zvec").replace("f(","self.get_m(")
        line = line.replace("xi","i").replace("yi","j").replace("zi","k")
        #line = line.replace("(","[").replace(")","]")
        lines.append(line)
        print(line)
    
    #print(lines)
    out = "{0}".format(cseFunc[1][0].transpose()[0,:].evalf())
    out = out.replace("xf","self.get_xvec").replace("yf","self.get_yvec").replace("zf","self.get_zvec").replace("f(","self.get_m(")
    out = out.replace("xi","i").replace("yi","j").replace("zi","k")
    print(out)
    
    def index1(i):
        str = ''
        if i == -1:
            str += 'm'
        if i == 0:
            str += 'z'
        if i == 1:
            str += 'p'
        if i == 2:
            str += 'P'
        return str
    
    def index2(i):
        str = ''
        if i == -1:
            str += 'm'
        if i == 0:
            str += 'z'
        if i == 1:
            str += 'p'
        if i == 2:
            str += 'P'
        return str
    
    def index(i,j,k):
        str = ''
        str += index1(i) + index1(j) + index1(k)
        return str

    vec = []

    i = 0
    while i <= 1:
        j = 0
        while j <= 1:
            k = 0
            while k <= 1:
                #f
                vec.append(symbols('f_{0}'.format(index(i,j,k))))
                
                #x10 = symbols('x_{0}'.format(index1(i))) - symbols('x_{0}'.format(index1(i-1)))
                #x02 = symbols('x_{0}'.format(index1(i+1))) - symbols('x_{0}'.format(index1(i)))
                #y10 = symbols('y_{0}'.format(index1(j))) - symbols('y_{0}'.format(index1(j-1)))
                #y02 = symbols('y_{0}'.format(index1(j+1))) - symbols('y_{0}'.format(index1(j)))
                #z10 = symbols('z_{0}'.format(index1(k))) - symbols('z_{0}'.format(index1(k-1)))
                #z02 = symbols('z_{0}'.format(index1(k+1))) - symbols('z_{0}'.format(index1(k)))
                
                x12 = symbols('x_{0}'.format(index1(i+1))) - symbols('x_{0}'.format(index1(i-1)))
                y12 = symbols('y_{0}'.format(index1(j+1))) - symbols('y_{0}'.format(index1(j-1)))
                z12 = symbols('z_{0}'.format(index1(k+1))) - symbols('z_{0}'.format(index1(k-1)))
                
                #fx,fy,fz
                #f0 = symbols('f_{0}'.format(index(i,j,k)))
                #fmzz = symbols('f_{0}'.format(index(i-1,j,k)))
                #fpzz = symbols('f_{0}'.format(index(i+1,j,k))) 
                #vec.append(((fmzz - f0)*x02 - (fpzz - f0)*x10)/(2*x10*x02))
                
                #fzmz = symbols('f_{0}'.format(index(i,j-1,k)))
                #fzpz = symbols('f_{0}'.format(index(i,j+1,k))) 
                #vec.append(((fzmz - f0)*y02 - (fzpz - f0)*y10)/(2*y10*y02))
                
                #fzzm = symbols('f_{0}'.format(index(i,j,k-1)))
                #fzzp = symbols('f_{0}'.format(index(i,j,k+1))) 
                #vec.append(((fzzm - f0)*z02 - (fzzp - f0)*z10)/(2*z10*z02))
                
                vec.append((symbols('f_{0}'.format(index(i+1,j,k))) - symbols('f_{0}'.format(index(i-1,j,k))) )/x12)
                vec.append((symbols('f_{0}'.format(index(i,j+1,k))) - symbols('f_{0}'.format(index(i,j-1,k))) )/y12)
                vec.append((symbols('f_{0}'.format(index(i,j,k+1))) - symbols('f_{0}'.format(index(i,j,k-1))) )/z12)

                #fxy,fxz,fyz
                vec.append((((symbols('f_{0}'.format(index(i+1,j+1,k))) - symbols('f_{0}'.format(index(i-1,j+1,k))) )/x12)-((symbols('f_{0}'.format(index(i+1,j-1,k))) - symbols('f_{0}'.format(index(i-1,j-1,k))) )/x12))/y12)
                vec.append((((symbols('f_{0}'.format(index(i+1,j,k+1))) - symbols('f_{0}'.format(index(i-1,j,k+1))) )/x12)-((symbols('f_{0}'.format(index(i+1,j,k-1))) - symbols('f_{0}'.format(index(i-1,j,k-1))) )/x12))/z12)
                vec.append((((symbols('f_{0}'.format(index(i,j+1,k+1))) - symbols('f_{0}'.format(index(i,j-1,k+1))) )/y12)-((symbols('f_{0}'.format(index(i,j+1,k-1))) - symbols('f_{0}'.format(index(i,j-1,k-1))) )/y12))/z12)
                
                #fxyz
                vec.append((((((symbols('f_{0}'.format(index(i+1,j+1,k+1))) - symbols('f_{0}'.format(index(i-1,j+1,k+1))) )/x12)-((symbols('f_{0}'.format(index(i+1,j-1,k+1))) - symbols('f_{0}'.format(index(i-1,j-1,k+1))) )/x12))/y12)-((((symbols('f_{0}'.format(index(i+1,j+1,k-1))) - symbols('f_{0}'.format(index(i-1,j+1,k-1))) )/x12)-((symbols('f_{0}'.format(index(i+1,j-1,k-1))) - symbols('f_{0}'.format(index(i-1,j-1,k-1))) )/x12))/y12))/z12)         

                k += 1
            j += 1
        i += 1
    vec = Matrix(vec)

    cseFunc = cse(vec,optimizations='basic')
    #generate the indices
    lines = ['im = i - 1','iz = i','ip = i + 1','iP = i + 2',
            'jm = j - 1','jz = j','jp = j + 1','jP = j + 2',
            'km = k - 1','kz = k','kp = k + 1','kP = k + 2']
    i = -1
    while i <= 2:
        j = -1
        while j <= 2:
            k = -1
            while k <= 2:
                var = index(i,j,k)
                line = "{0} = self.index(i{1},j{2},k{3})".format(index(i,j,k),index2(i),index2(j),index2(k))
                lines.append(line)
                k += 1
            j += 1
        i += 1
    def replaceIndices(f):
        i = -1
        while i <= 2:
            j = -1
            while j <= 2:
                k = -1
                while k <= 2:
                    var = index(i,j,k)
                    f = f.replace(var,'[{0}]'.format(var))
                    k += 1
                j += 1
            i += 1
        return f
    def replaceIndices2(f):
        f = f.replace('x_m','self.xvec[im]')
        f = f.replace('x_z','self.xvec[iz]')
        f = f.replace('x_p','self.xvec[ip]')
        f = f.replace('x_P','self.xvec[iP]')
        f = f.replace('y_m','self.yvec[jm]')
        f = f.replace('y_z','self.yvec[jz]')
        f = f.replace('y_p','self.yvec[jp]')
        f = f.replace('y_P','self.yvec[jP]')
        f = f.replace('z_m','self.zvec[km]')
        f = f.replace('z_z','self.zvec[kz]')
        f = f.replace('z_p','self.zvec[kp]')
        f = f.replace('z_P','self.zvec[kP]')
        return f
        
    for expr in cseFunc[0]:
        f = str(expr[1])
        f = replaceIndices(f)
        f = replaceIndices2(f)
        f = f.replace('f_','self.m')
        line = '{0} = {1}'.format(expr[0], f)
        lines.append(line)
        
    bvec = str(cseFunc[1][0].transpose())
    bvec = bvec.replace('Matrix([','bvec = np.array(')
    bvec = bvec.replace('])',')')
    bvec = bvec.replace(',',',\n')
    bvec = replaceIndices(bvec)
    bvec = replaceIndices2(bvec)
    bvec = bvec.replace('f_','self.m')
    lines.append(bvec)
    code = ''
    for line in lines:
        code += line+'\n'
    print(code)

if __name__=='__main__':
    genFuncCalls()
    #generateBinv()
    #optimizeBvecFormation()
    #testResult()


# In[52]:

from sympy import *

x,y,z = symbols('x y z')
f = Function('f')
f1 = f(x+1,y,z)
f2 = f(f1.args[0] +1,y,z) + f1
e = f1.func
h = f1 - f(x-3,y,z)
g = Function('g')
g = f1 + f(x,y,z)


# In[54]:

g.subs({'x':x+6})


# In[116]:

a,o = symbols('a o')
from sympy.solvers import solve
ne = a*exp(-x**2/(o*log(2)*Rational(7,2))**2)
c = cse(ne.diff(x,x,x,x,x).subs({o:10,a:1e13}))
print(c)
for k in c[0]:
    print("{0} = {1}".format(k[0],k[1]))


# In[102]:

import numpy as np
f=lambdify(x,ne.diff(x,x,x,x,x).subs({o:10,a:1e13}),"numpy")
import pylab as plt
plt.plot(np.linspace(0,100,1000),f(np.linspace(0,100,1000)))
plt.show()
ne.diff(x,x,x,x,x).subs({x:o})


# In[46]:

def genFuncCalls():
    from sympy import symbols, Rational, Matrix,Function
    from sympy.vector import Vector
    from sympy import cse

    x,y,z = symbols('x y z')
    a = Function('a')
    wholeFunc = []
    alphaVec = []
    i = 0
    while i <= 3:
        j = 0
        while j <= 3:
            k = 0
            while k <= 3:
                afunc = a(Rational(k)+Rational(4)*(Rational(j) + Rational(4)*Rational(i)))
                alphaVec.append(afunc)
                wholeFunc.append(x**Rational(i)*y**Rational(j)*z**Rational(k))
                k += 1
            j += 1
        i += 1
    
    alphaVec = Matrix(alphaVec)
    wholeFunc = Matrix(wholeFunc)
    cornerVec = Matrix([wholeFunc.transpose()[:],
                 wholeFunc.diff(x).transpose()[:],
                 wholeFunc.diff(y).transpose()[:],
                 wholeFunc.diff(z).transpose()[:],
                 wholeFunc.diff(x,y).transpose()[:],
                 wholeFunc.diff(x,z).transpose()[:],
                 wholeFunc.diff(y,z).transpose()[:],
                 wholeFunc.diff(x,y,z).transpose()[:]])
    
    cseFunc = cse(cornerVec,optimizations='basic')
    #print(cornerVec.shape,alphaVec.shape)
    #print(Matrix(cornerVec.dot(alphaVec)))
    
    #print (cseFunc)
    lines = []
    for optLine in cseFunc[0]:
        line = "{0} = {1}".format(optLine[0],optLine[1])
        #line = line.replace("xf","self.get_xvec").replace("yf","self.get_yvec").replace("zf","self.get_zvec").replace("f(","self.get_m(")
        #line = line.replace("xi","i").replace("yi","j").replace("zi","k")
        #line = line.replace("(","[").replace(")","]")
        lines.append(line)
        print(line)
    
    #print(lines)
    out = "np.array({0},dtype=np.double)".format(cseFunc[1][0])
    #out = out.replace("xf","self.get_xvec").replace("yf","self.get_yvec").replace("zf","self.get_zvec").replace("f(","self.get_m(")
    #out = out.replace("xi","i").replace("yi","j").replace("zi","k")
    print(out)
    
    

genFuncCalls()


# In[ ]:




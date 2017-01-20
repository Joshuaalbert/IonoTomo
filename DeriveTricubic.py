
# coding: utf-8

# In[9]:

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
                a = symbols('a_{0}{1}{2}'.format(i,j,k))
                alphaVec.append(a)
                wholeFunc += a*x**Rational(i)*y**Rational(j)*z**Rational(k)
                k += 1
            j += 1
        i += 1

    #print(alphaVec)

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
    
def optimizeBvecFormation():
    from sympy import symbols, Matrix
    from sympy import cse
    
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
    generateBinv()
    optimizeBvecFormation()
    #testResult()


# In[ ]:




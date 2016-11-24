
# coding: utf-8

# In[16]:

'''Contains geometry algorithms'''
import numpy as np
from itertools import combinations

## Use instead of zero. 1e4 * machine_precision
epsFloat = (7/3. - 4/3. - 1)*1e3


###
# Rotation matrices
###

def rot(dir,theta):
    '''create the rotation matric about unit vector dir by theta radians.
    Appear anticlockwise when dir points toward observer.
    and clockwise when pointed away from observer. 
    **careful when visualizing with hands**'''
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,-dir[2]*s,dir[1]*s],
                     [dir[2]*s,c,-dir[0]*s],
                     [-dir[1]*s,dir[0]*s,c]]) + (1-c)*np.outer(dir,dir)

def rotx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1,0,0],
                    [0,c,-s],
                    [0,s,c]])
def roty(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,0,s],
                    [0,1,0],
                    [-s,0,c]])

def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,-s,0],
                    [s,c,0],
                    [0,0,1]])

def rotAxis(R):
    '''find axis of rotation for R using that (R-I).u=(R-R^T).u=0'''
    u = np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    #|u| = 2 sin theta
    return u/np.linalg.norm(u)
    
def rotAngle(R):
    '''Given a rotation matix R find the angle of rotation.
    Use that Tr(R) = 1 + 2 cos(theta)'''
    return np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.)/2.)

###
# UVW coordinates
###

def localENU2uvw(enu,alt,az,lat):
    '''Given a local ENU vector, rotate to uvw coordinates'''
    #first get hour angle and dec from alt az and lat
    ha,dec = altAz2hourangledec(alt,az,lat)
    return rotx(lat).dot(rotz(-ha)).dot(rotx(lat-dec)).dot(enu)

def itrs2uvw(ha,dec,lon,lat):
    #ha,dec = altAz2hourangledec(alt,az,lat)
    R = rotx(-lat).dot(rotz(-ha)).dot(rotx(np.pi/2.-dec)).dot(rotz(np.pi/2. + lon))
    udir = R.dot(np.array([1,0,0]))
    vdir = R.dot(np.array([0,1,0]))
    wdir = R.dot(np.array([0,0,1]))
    return np.array([udir,vdir,wdir])
    udir = rotz(lon-ha).dot(roty(- dec - lat)).dot(np.array([0,1,0]))
    vdir = rotz(lon-ha).dot(roty(- dec - lat)).dot(np.array([0,0,1]))
    wdir = rotz(lon-ha).dot(roty(- dec - lat)).dot(np.array([1,0,0]))
    #print np.linalg.norm(udir),np.linalg.norm(vdir),np.linalg.norm(wdir)
    return np.array([udir,vdir,wdir])

###
# Miscellaneous routines
###

def gramSchmidt(dir):
    '''Get an ortho system of axes'''
    raxis = np.cross(dir,np.array([0,0,1]))
    mag = np.linalg.norm(raxis) 
    if mag == 0:
        zdir = np.array([0,0,1])
        xdir = np.array([1,0,0])
        ydir = np.array([0,1,0])
    else:
        R = rot(raxis,np.arcsin(mag/np.linalg.norm(dir)))
        zdir = dir
        xdir = (np.eye(3) - np.outer(zdir,zdir)).dot(R.dot(np.array([1,0,0])))
        xdir /= np.linalg.norm(xdir)
        ydir = (np.eye(3) - np.outer(zdir,zdir)- np.outer(xdir,xdir)).dot(R.dot(np.array([0,1,0])))
        ydir /= np.linalg.norm(ydir)
    return xdir,ydir,zdir    

###
# Geometric objects
###

class Ray(object):
    def __init__(self,origin,direction,id=-1):
        if id >= 0:
            self.id = id
        self.origin = np.array(origin)
        self.dir = np.array(direction)/np.sqrt(np.dot(direction,direction))
    def eval(self,t):
        '''Return the location along the line'''
        return self.origin + t*self.dir
    def __repr__(self):
        return "Ray: origin: {0} -> dir: {1}".format(self.origin,self.dir)
    
class LineSegment(Ray):
    def __init__(self,p1,p2,id=-1):
        c=np.array(p2)-np.array(p1)
        self.sep = np.linalg.norm(c)
        super(LineSegment,self).__init__(p1,c,id=id)
        #if np.alltrue(p1==p2):
         #
            #print("Point not lineSegment")        
    def __repr__(self):
        return "LineSegment: p1: {0} -> p2 {1}, sep: {2}".format(self.eval(0),self.eval(self.sep),self.sep)

###
# Line routines
###

def inBounds(seg,t):
        return (t>0) and (t <= seg.sep)

def midPointLineSeg(seg):
    '''Return midpoint of linesegment'''
    return seg.eval(seg.sep/2.)

class Plane(object):
    def __init__(self,p1,p2=None,p3=None,p4=None,normal=None):
        '''If normal is defined and surface defines a part of a convex polyhedron,
        take normal to point inside the poly hedron.
        ax+by+cz+d=0'''
        if normal is None:
            if p2 is None and p3 is None:
                print("Not enough information")
                return
            #normal is in direction of p1p2 x p1p3
            a = p1[1]*(p2[2] - p3[2]) + p2[1]*(p3[2] - p1[2]) + p3[1]*(p1[2] - p2[2])
            b = p1[2]*(p2[0] - p3[0]) + p2[2]*(p3[0] - p1[0]) + p3[2]*(p1[0] - p2[0])
            c = p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])
            d = -p1[0]*(p2[1]*p3[2] - p3[1]*p2[2]) - p2[0]*(p3[1]*p1[2] - p1[1]*p3[2]) - p3[0]*(p1[1]*p2[2] - p2[1]*p1[2])
            self.n = np.array([a,b,c])
            self.d = d/np.linalg.norm(self.n)
            self.n = self.n/np.linalg.norm(self.n)
        else:
            self.n = np.array(normal)
            self.n = self.n/np.linalg.norm(self.n)
            self.d = -self.n.dot(p1)
    def __repr__(self):
        return "Plane: normal: {0}, d=-n.p {1}".format(self.n,self.d)
###
# Functions on planes
###

def normalSide(plane,p):
    '''Return whether point p is on the normal side of the plane or None if on the plane.'''
    s = plane.n.dot(p) + plane.d
    if s > epsFloat:
        return True
    if s < -epsFloat:
        return False
    return None

def onPlane(plane,p):
    '''Return whether point p is on the plane within epsFloat.'''
    a = np.abs(plane.n.dot(p) + plane.d)
    if a < epsFloat:
        return True
    else:
        return False
        #print a,epsFloat 
    
def coplanarPoints(points):
    '''Return'''
    if len(points) < 3:
        #print ("Not enough points to test")
        return False,None
    if len(points) == 3:
        return True, Plane(*points)
    plane = Plane(*points[:3])
    i = 3
    while i < len(points):
        if not onPlane(plane,points[i]):
            return False, None
        i += 1
    return True,plane



class BoundedPlane(Plane):
    '''assumes convex hull of 4 points'''
    def __init__(self,vertices):
        if len(vertices) != 4:
            #print("Not enough vertices")
            return
        res,plane = coplanarPoints(vertices)
        if res:
            super(BoundedPlane,self).__init__(*vertices)
            
            self.centroid = np.mean(vertices,axis=0)
            #first triangle
            edge01 = LineSegment(vertices[0],vertices[1])
            edge12 = LineSegment(vertices[1],vertices[2])
            edge20 = LineSegment(vertices[2],vertices[0])
            
            centroid1 = (vertices[0] + vertices[1] + vertices[2])/3.
            centerDist = np.linalg.norm(centroid1 - vertices[3])
            if np.linalg.norm(midPointLineSeg(edge01) - vertices[3]) < centerDist:
                #reject 01
                self.edges = [edge12,edge20,LineSegment(vertices[0],vertices[3]),
                         LineSegment(vertices[1],vertices[3])]
            if np.linalg.norm(midPointLineSeg(edge12) - vertices[3]) < centerDist:
                #reject 12
                self.edges = [edge01,edge20,LineSegment(vertices[1],vertices[3]),
                         LineSegment(vertices[2],vertices[3])]
            if np.linalg.norm(midPointLineSeg(edge20) - vertices[3]) < centerDist:
                #reject 20
                self.edges = [edge01,edge12,LineSegment(vertices[2],vertices[3]),
                         LineSegment(vertices[0],vertices[3])]   
        else:
            print('not coplanar',vertices)

    def __repr__(self):
        return "Bounded Plane: edges: {0}, n {1}".format(self.edges,self.n)
    

        
###
# Functions between geometric objects
###
                    
def projLineSegPlane(line,plane):
    '''Project a lineseg onto a plane. i.e. get the components in the plane.'''
    proj = np.eye(np.size(line.origin)) - np.outer(plane.n,plane.n)
    x1 = proj.dot(line.origin)
    x2 = proj.dot(line.eval(line.sep))
    return LineSegment(x1,x2)

def projRayPlane(line,plane):
    '''Project a ray onto a plane.'''
    proj = np.eye(np.size(line.origin)) - np.outer(plane.n,plane.n)
    x1 = proj.dot(line.origin)
    dir = proj.dot(line.dir)
    return Ray(x1,dir)    

def intersectPointRay(point,ray):
    '''Get whether a point is on a line and the point
    or the shortest line segment connecting the them.'''
    diff = point - ray.origin
    t = diff.dot(ray.dir)
    p = ray.eval(t)
    if diff.dot(diff) - t**2 > epsFloat:
        return False, LineSegment(point,p)
    else:
        return True, p
    
def intersectRayRay(ray1,ray2):
    '''Return whether ray1 and ray2 intersect and the point,
    or the shortest linesegment connecting te two.
    '''
    n1n2 = ray1.dir.dot(ray2.dir)
    B = ray1.origin - ray2.origin
    det = n1n2*n1n2 - 1
    if det < epsFloat:
        #print("ray1 and ray2 are parallel")
        t1 = -B.dot(ray1.dir)
        t2 = 0
    else:
        dPn1 = B.dot(ray1.dir)
        dPn2 = B.dot(ray2.dir)
        t1 = (-n1n2*dPn2 + dPn1)/det
        t2 = (n1n2*dPn1 - dPn2)/det
    p1 = ray1.eval(t1)
    p2 = ray2.eval(t2)
    d = distPointPoint(p1,p2)
    if (d < epsFloat):
        return True, p1
    else:
        return False,LineSegment(p1,p2)
    
def intersectRayPlane(ray,plane,positiveOnly = False, entryOnly=False, exitOnly=False):
    '''Determine if ray hits plane. 
    Return whether it does and the point if so. 
    positiveOnly only returns true if plane is casually in front of ray origin (t>0)
    entryOnly and exitOnly assumes normal points inwards of polygon.
    ''' 
    parallel = ray.dir.dot(plane.n)#3m 3a
    if parallel*parallel < epsFloat*epsFloat:#2m 1logic
        return False,None
    else:
        if entryOnly:#1 logic
            if parallel < epsFloat:#1m 1logic
                return False,None# going exit
        if exitOnly:# logic
            if parallel > -epsFloat:#1m 1logic
                return False,None# going entry
    #make point
    c0p0 = ray.origin - plane.centroid#3a
    t = -(c0p0.dot(plane.n)/parallel)#5m 3a
    if positiveOnly:#1 logic
        if t < -epsFloat:#1logic, casual
            return False,None
    x = ray.origin + t*ray.dir #3m 3a
    return True,x#13m 12a 3(5)logic +6 assign = 156flop
    
def intersectPlanePlane(plane1,plane2):
    '''calculate intersection of 2 planes  which is one ray or None.'''

    n1n2 = plane1.n.dot(plane2.n)#3 + 3
    if n1n2 > 1 - epsFloat:
        #print ("Parallel planes")
        return False, None
    n1n1 = plane1.n.dot(plane1.n)#3 + 3
    n2n2 = plane2.n.dot(plane2.n)#3 + 3
    det = n1n1*n2n2 - n1n2*n1n2#2 + 1
    c1 = (plane2.d*n1n2 - plane1.d*n2n2)/det#3 + 1
    c2 = (plane1.d*n1n2 - plane2.d*n1n1)/det#3 + 1
    u = np.cross(plane1.n,plane2.n)#6 + 3
    ray = Ray(c1*plane1.n + c2*plane2.n,u)#6 + 3
    return True,ray#35 + 18
    
    u = np.cross(plane1.n,plane2.n)#6 + 3
    uMag = np.linalg.norm(u)#
    if uMag < epsFloat:
        # print ("Parallel planes")
        return False,None
    u /= uMag
    i = np.argmax(u)
    den = u[i]
    if i == 0:
        ray = Ray(np.array([0,(plane1.n[2]*plane2.d - plane2.n[2]*plane1.d)/den,
                            (plane2.n[1]*plane1.d - plane1.n[1]*plane2.d )/den]),u)#6 + 2
        return True,ray#12 + 5
    if i == 1:
        ray = Ray(np.array([(plane1.n[2]*plane2.d - plane2.n[2]*plane1.d)/den,0,
                            (plane2.n[0]*plane1.d - plane1.n[0]*plane2.d )/den]),u)
        return True,ray
    if i == 2:
        ray = Ray(np.array([(plane1.n[1]*plane2.d - plane2.n[1]*plane1.d)/den,
                            (plane2.n[0]*plane1.d - plane1.n[0]*plane2.d )/den,0]),u)
        return True,ray

def intersectPlanePlanePlane(plane1,plane2,plane3):
    '''Return intersection of three planes which forms a point'''
    n23 = np.cross(plane2.n,plane3.n)
    n31 = np.cross(plane3.n,plane1.n)
    n12 = np.cross(plane1.n,plane2.n)
    n123 = plane1.n.dot(n23)
    if (n123 == 0):
        return False, None
    else:
        return True, -(plane1.d*n23 + plane2.d*n31 + plane3.d*n12)/n123

def intersectRayBoundedPlaneHull(ray,plane,positiveOnly = False, entryOnly=False, exitOnly=False):
    '''Determine if ray hits hull. 
    Return whether it does and the point if so. 
    positiveOnly only returns true if plane is casually in front of ray origin (t>0)
    ''' 
    parallel = ray.dir.dot(plane.n)#3m 3a
    if parallel*parallel < epsFloat*epsFloat:#2m 1logic
        return False,None
    else:
        if entryOnly:#1 logic
            if parallel < epsFloat:#1m 1logic
                return False,None# going exit
        if exitOnly:# logic
            if parallel > -epsFloat:#1m 1logic
                return False,None# going entry
    #make point
    c0p0 = ray.origin - plane.centroid#3a
    t = -(c0p0.dot(plane.n)/parallel)#5m 3a
    if positiveOnly:#1 logic
        if t < -epsFloat:#1logic, casual
            return False,None
    x = ray.origin + t*ray.dir #3m 3a
    #it hits and check boundedness
    for edge in plane.edges:#x4
        r0x = x - edge.origin#3a
        r0c0 = plane.centroid - edge.origin#3a
        if r0x.dot(r0c0) - r0x.dot(edge.dir)*r0c0.dot(edge.dir) < -epsFloat:#10m 13a 1logic
            return False, None
    return True,x#53m 88a 7(9)logic +6 assign = 156flop

def planesOfCuboid(center,dx,dy,dz):
    '''return bounding planes of cuboid. 
    
    Legacy. Use boundPlanesOfCuboid!'''
    planes = []
    planes.append(Plane(np.array(center) - np.array([dx/2.,0,0]),normal=np.array([1,0,0])))
    planes.append(Plane(np.array(center) - np.array([-dx/2.,0,0]),normal=np.array([-1,0,0])))
    planes.append(Plane(np.array(center) - np.array([0,dy/2.,0]),normal=np.array([0,1,0])))
    planes.append(Plane(np.array(center) - np.array([0,-dy/2.0,0]),normal=np.array([0,-1,0])))
    planes.append(Plane(np.array(center) - np.array([0,0,dz/2.]),normal=np.array([0,0,1])))
    planes.append(Plane(np.array(center) - np.array([0,0,-dz/2.]),normal=np.array([0,0,-1])))
    return planes


def boundPlanesOfCuboid(center,dx,dy,dz):
    '''
    Return bounding planes of cuboid centered at center, with sides of dx,dy,dz. 
    The ordering of the planes is important for relative position in OctTree.
*        +-----+
*        ^  3  |
*        z  N. |
*  +--<z-+--x>-#--z>-+-<x--+
*  |  W. ^  D. | E.  | U.  |
*  |  0  y  4  |  1  |  5  |
*  +-----O--x>-+-----+-----+
*        z  S. |
*        v  2  |
*        +-----+
*  
    '''
    planes = []    
    #West - x
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,-dz/2.])]))
    #East + x
    planes.append(BoundedPlane([np.array(center) + np.array([dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,-dz/2.])]))
    #South - y
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([-dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,dz/2.])]))
    #North + y
    planes.append(BoundedPlane([np.array(center) + np.array([dx/2.,dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,-dz/2.])]))
    #Down - z
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,-dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,-dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,-dz/2.])]))
    #Up +z
    planes.append(BoundedPlane([np.array(center) + np.array([-dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,dy/2.,dz/2.]),
                                np.array(center) + np.array([dx/2.,-dy/2.,dz/2.]),
                                np.array(center) + np.array([-dx/2.,dy/2.,dz/2.])]))
    return planes
        
class Voxel(object):
    def __init__(self,center=None,dx=None,dy=None,dz=None,boundingPlanes=None):
        '''Create a volume out of bounding planes.'''
        boundingPlanes = boundPlanesOfCuboid(center,dx,dy,dz)
        if len(boundingPlanes)!=6:
            print ("Failed to make bounding planes for center {0}, dx {1}, dy {2}, dz {3}, planes {4}".format(center,dx,dy,dz,len(boundingPlanes)))
            
        if boundingPlanes is not None:
            self.vertices = []
            planeTriplets = combinations(boundingPlanes,3)
            for triplet in planeTriplets:
                
                res,point = intersectPlanePlanePlane(*triplet)
                if res:
                    self.vertices.append(point)
            if len(self.vertices)<8:
                print("Planes don't form voxel",len(self.vertices))
                self.volume = 0
                return
            self.centroid = np.mean(self.vertices,axis=0)
            sides = np.max(self.vertices,axis=0) - np.min(self.vertices,axis=0)
            self.dx = sides[0]
            self.dy = sides[1]
            self.dz = sides[2]
            self.volume = self.dx*self.dy*self.dz
        self.boundingPlanes = boundingPlanes         
    def __repr__(self):
        return "Voxel: Center {0}\nVertices:\t{2}".format(self.centroid,self.vertices)

###
# OctTree object and routines below.
# OctTree partitions 3d in 8 self-similar children
###
class OctTree(Voxel):
    def __init__(self,center=None,dx=None,dy=None,dz=None,boundingPlanes=None,parent=None,properties=None,id=-1):
        super(OctTree,self).__init__(center,dx,dy,dz,boundingPlanes)
        self.id = id
        self.parent = parent
        self.children = []
        self.hasChildren = False
        self.lineSegments = {}
        #properties are extensive or intensive.
        #if extensive then the total property is the sum of subsystems' property
        #if intensive then the total property is the average of the subsystems' property
        #values are mean and standard deviation
        if properties is not None:
            self.properties = {}
            for key in properties.keys():
                if properties[key][0] == 'intensive':
                    self.properties[key] = ['intensive',properties[key][1],properties[key][2]]
                elif properties[key][0] == 'extensive':
                    self.properties[key] = ['extensive',properties[key][1],properties[key][2]]
        else:
            self.properties = {'n':['intensive',1,0.01]}#'Ne':['extensive',0,1]
        self.lineSegments = {}
   
    def __repr__(self):
        return "OctTree: center {0} hasChildren {1}".format(self.centroid,self.hasChildren)  
    
###
# OctTree routines
###

def accumulateChildren(octTree):
    '''Accumulate the properties of children up through te octTree.
    Should be done before killing children in any node.'''
    if octTree.hasChildren:
        for key in octTree.properties.keys():
            octTree.properties[key][1] = 0
            octTree.properties[key][2] = 0
        for child in octTree.children:
            accumulateChildren(child)
            for key in self.properties.keys():
                if octTree.properties[key][0] == 'intensive':
                    octTree.properties[key][1] += child.volume*child.properties[key][1]
                    octTree.properties[key][2] += (child.volume*child.properties[key][2])**2
                elif octTree.properties[key][0] == 'extensive':
                    octTree.properties[key][1] += child.properties[key][1]
                    octTree.properties[key][2] += child.properties[key][2]**2
        for key in octTree.properties.keys():
            if octTree.properties[key][0] == 'intensive':
                octTree.properties[key][1] /= octTree.volume
                octTree.properties[key][2] = np.sqrt(octTree.properties[key][2])/octTree.volume
            if octTree.properties[key][0] == 'extensive':
                octTree.properties[key][2] = np.sqrt(octTree.properties[key][2])

def killChildren(octTree,takeProperties=True):
    '''Delete all lower branches if they exist.
    Take properties from children if required'''

    if takeProperties:
        accumulateChildren(octTree)
    if octTree.hasChildren:    
        #remove the reference (python automatically cleans up when no more reference)
        del octTree.children
        octTree.hasChildren = False

def getAllBoundingPlanes(octTree):
    '''Accumulate all bounding planes. 6 per vox!
    Expensive in memory and time.
    8x longer per depth!'''
    boundingPlanes = []
    if octTree.hasChildren:
        for child in octTree.children:
            boundingPlanes = boundingPlanes + getAllBoundingPlanes(child)
        return boundingPlanes
    else:
        return octTree.boundingPlanes

def getAllDecendants(octTree):
    '''Get all lowest chilren on tree.'''
    out = []
    if octTree.hasChildren:
        for child in octTree.children:
            out = out + getAllDecendants(child)
        return out
    else:
        return [octTree]
    
def countDecendants(octTree):
    '''Count number of lowest children.
    8x longer per layer'''
    if octTree.hasChildren:
        sum = 0
        for child in octTree.children:
            sum += countDecendants(child)
        return sum
    else:
        return 1

def minCellSize(octTree):
    '''Recursive discovery of the smallest cell size. 
    8x Expensive!'''
    if octTree.hasChildren:
        cellSizes = []
        for child in octTree.children:
            cellSizes.append(minCellSize(child))
        minAx = np.min(cellSizes,axis=0)
        return minAx
    else:
        return np.array([octTree.dx,octTree.dy,octTree.dz])

def getDepth(octTree,childDepth=0):
    '''Get depth of OctTree by pushing through parents'''
    if octTree.parent is None:
        return childDepth
    else:
        return getDepth(octTree.parent,childDepth+1)

def intersectRay(octTree,ray):
    '''determine if the ray hits the exterior of an octTree in forward direction.'''
    i = 0
    while i < 6:
        plane = octTree.boundingPlanes[i]
        res,point = intersectRayBoundedPlaneHull(ray,plane,positiveOnly=True,entryOnly=True)
        if res:#ray entering from outside or on boundary, not from within
            #print("Hit entry plane:",plane,"in vox:",self)
            return True, point, i
        i += 1
    return False, None, None

def propagateRay(octTree,ray):
    '''Propagate ray until it leaves polygon. Can fail if polygon has a hole in it.'''
    i = 0
    while i < 6:
        plane = octTree.boundingPlanes[i]
        res,point1 = intersectRayPlane(ray,plane,positiveOnly=True,exitOnly=True)
        if res:
            d1 = point1 - ray.origin
            distwin = d1.dot(d1)
            iwin = i
            pointwin = point1
            while i < 6:
                plane = octTree.boundingPlanes[i]
                res,point2 = intersectRayPlane(ray,plane,positiveOnly=True,exitOnly=True)
                if res:
                    d2 = point2 - ray.origin
                    dist2 = d2.dot(d2)
                    i2 = i
                    pointwin = point2
                    if dist2 < distwin:
                        iwin = i2
                        distwin = dist2
                    while i < 6:
                        plane = octTree.boundingPlanes[i]
                        res,point3 = intersectRayPlane(ray,plane,positiveOnly=True,exitOnly=True)
                        if res:
                            d3 = point3 - ray.origin
                            dist3 = d3.dot(d3)
                            i3 = i
                            pointwin = point3
                            if dist3 < distwin:
                                iwin = i3
                                distwin = dist3
                        i += 1
                i += 1
        i += 1    
    return True,pointwin,iwin

def intersectPoint(octTree,point):
    '''Return octtree at lowest level.'''
    if octTree.hasChildren:
        quad = (point[0] > octTree.centroid[0]) + 2*(point[1] > octTree.centroid[1]) + 4*(point[2] > octTree.centroid[2])
        return intersectPoint(octTree.children[quad],point)
    else:
        return octTree

def getOtherSide(octTree,planeIdx):
    '''
* 3 bits (a,b,c)
*           Top
*        +-----+-----+
*        ^  6  |  7  |
*        y N.W.| N.E.|
*        +-----O-x>--+
*        | S.W.| S.E.|
*        |  4  |  5  |
*        +-----+-----+
*           Bottom
*        +-----+-----+
*        ^  2  |  3  |
*        y N.W.| N.E.|
*        +-----O-x>--+
*        | S.W.| S.E.|
*        |  0  |  1  |
*        +-----+-----+
* 3 qubits (a1,a2),(b1,b2),(c1,c2)
*        +-----+
*        ^  3  |
*        z  N. |
*  +--<z-+--x>-#--z>-+-<x--+
*  |  W. ^  D. | E.  | U.  |
*  |  0  y  4  |  1  |  5  |
*  +-----O--x>-+-----+-----+
*        z  S. |
*        v  2  |
*        +-----+
    '''

    childEW = (octTree.id >> 0) & 1#b0 childEW = (self.id & 1) >> 0
    childNS = (octTree.id >> 1) & 1#b1 childNS = (self.id & 2) >> 1
    childUD = (octTree.id >> 2) & 1#b2 childUD = (self.id & 4) >> 2
    #plane on same side as quadrant, p=2^planeIdx == 1<<(bi + 2*i)
    #or planeIdx == bi + 2*i
    #print("Cube: EW {0} NS {1} UD {2}".format(childEW,childNS,childUD))
    #print("Plane: ",planeIdx)
    if (childEW == planeIdx) or (childNS + 2 == planeIdx) or (childUD + 4 == planeIdx):
        return -(planeIdx+1)
    if (~childEW + 2) == planeIdx:#flip EW
        otherEW = (~childEW + 2)
    else:
        otherEW = childEW
    if (~childNS + 4) == planeIdx:#flip NS
        otherNS = (~childNS + 2)
    else:
        otherNS = childNS
    if (~childUD + 6) == planeIdx:#flip UD
        otherUD = (~childUD + 2)
    else:
        otherUD = childUD
    return (otherEW << 0) + (otherNS << 1) + (otherUD << 2)

        
def subDivide(octTree):
    '''Make eight voxels to partition this voxel. 
    Takes 8x longer per depth!
*           Top
*        +-----+-----+
*        ^  6  |  7  |
*        y N.W.| N.E.|
*        +-----O-x>--+
*        | S.W.| S.E.|
*        |  4  |  5  |
*        +-----+-----+
*           Bottom
*        +-----+-----+
*        ^  2  |  3  |
*        y N.W.| N.E.|
*        +-----O-x>--+
*        | S.W.| S.E.|
*        |  0  |  1  |
*        +-----+-----+
'''
    
    if octTree.hasChildren:
        for child in octTree.children:
            subDivide(child)
    else:
        octTree.children = []
        octTree.hasChildren = True
        dx = octTree.dx/2.
        dy = octTree.dy/2.
        dz = octTree.dz/2.
        #000 BSW
        octTree.children.append(OctTree(octTree.centroid - np.array([dx/2.,dy/2.,dz/2.]),dx,dy,dz,parent=octTree,id=0))
        #001 BSE
        octTree.children.append(OctTree(octTree.centroid - np.array([-dx/2.,dy/2.,dz/2.]),dx,dy,dz,parent=octTree,id=1))
        #010 BNW
        octTree.children.append(OctTree(octTree.centroid - np.array([dx/2.,-dy/2.,dz/2.]),dx,dy,dz,parent=octTree,id=2))
        #011 BNE
        octTree.children.append(OctTree(octTree.centroid - np.array([-dx/2.,-dy/2.,dz/2.]),dx,dy,dz,parent=octTree,id=3))
        #100 TSW
        octTree.children.append(OctTree(octTree.centroid - np.array([dx/2.,dy/2.,-dz/2.]),dx,dy,dz,parent=octTree,id=4))
        #101 TSE
        octTree.children.append(OctTree(octTree.centroid - np.array([-dx/2.,dy/2.,-dz/2.]),dx,dy,dz,parent=octTree,id=5))
        #110 TNW
        octTree.children.append(OctTree(octTree.centroid - np.array([dx/2.,-dy/2.,-dz/2.]),dx,dy,dz,parent=octTree,id=6))
        #111 TNE
        octTree.children.append(OctTree(octTree.centroid - np.array([-dx/2.,-dy/2.,-dz/2.]),dx,dy,dz,parent=octTree,id=7))
    return octTree       

def subDivideToDepth(octTree,depth):
    '''Do depth subDivides. No max depth checking.'''
    i = 0
    while i < depth:
        subDivide(octTree)
        i += 1
    return octTree
    
def saveOctTree(fileName, octTree):
    try:
        np.save(fileName,octTree,fix_imports=True)
    except:
        np.save(fileName,octTree)

def loadOctTree(fileName):
    try:
        return np.load(fileName,fix_imports=True).item(0)
    except:
        return np.load(fileName).item(0)
    
def snellsLaw(n1,n2,ray,normal,point):
    '''Produce a ray following snells law at an interface with normal incidence pointing out.'''
    #snells law here
    axis = np.cross(-normal,ray.dir)
    sintheta1 = np.linalg.norm(axis)
    dTheta = np.arcsin(n1/n2*sintheta1*(np.sqrt(1-sintheta1**2) - np.sqrt((n2/n1)**2-sintheta1**2)))
    #print (dTheta)
    ## or
    #dTheta = np.arcsin(n1/n2*sintheta1) - np.arcsin(sintheta1)
    #print(dTheta)
    dir2 = rot(axis,dTheta).dot(ray.dir)
    #todo
    propRay = Ray(point,dir2,id=ray.id)
    return propRay
    
def forwardRay(ray,octTree):
    '''Propagate ray through octTree until it leaves the boundaries'''
    #from outside to octTree
    inside,entryPoint,entryPlaneIdx = intersectRay(octTree,ray)
    if not inside:
        print('failed to hit',octTree)
        return
    vox = intersectPoint(octTree,entryPoint)
    normal = -vox.boundingPlanes[entryPlaneIdx].n
    n1 = 1
    entryRay = ray
    while inside:
        n2 = vox.properties['n'][1]
        
        rayProp = snellsLaw(n1,n2,entryRay,normal,entryPoint)
        res,exitPoint,exitPlaneIdx = propagateRay(vox,rayProp)
        if not res:
            print("something went wrong and ",rayProp," didn't exit ",vox)
            return

        vox.lineSegments[rayProp.id] = LineSegment(entryPoint,exitPoint)#np.linalg.norm(exitPoint-entryPoint)

        #print(vox.lineSegments)
        #resolve the next vox to hit
        this = vox
        unresolved = True
        while unresolved:
            if this.parent is None:
                inside = False
                break
                
            nextVoxIdx = getOtherSide(this,exitPlaneIdx)
            if nextVoxIdx >= 0:#hits a sibling of this
                nextVox = this.parent.children[nextVoxIdx]
                #res,entryPoint,entryPlaneIdx = intersectRay(nextVox,rayProp)
                #if not res:
                #    print("failed to hit sibling",nextVox)
                #    return
                entryPoint = exitPoint
                vox = intersectPoint(nextVox,entryPoint)
                normal = this.boundingPlanes[exitPlaneIdx].n
                n1 = n2
                entryRay = rayProp
                unresolved = False
                continue
            else:#nextVoxIdx is -planeIdx-1 of parent
                this = this.parent
                exitPlaneIdx = -(nextVoxIdx+1)
    return exitPoint,vox.boundingPlanes[exitPlaneIdx]
        
def plotOctTreeXZ(octTree,ax=None):
    import pylab as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        ax = plt.subplot(111)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    voxels = getAllDecendants(octTree)
    for vox in voxels:
        #plot S plane (2)
        for edge in vox.boundingPlanes[3].edges:
            p1 = edge.origin
            p2 = edge.eval(edge.sep)
            ax.plot([p1[0],p2[0]],[p1[2],p2[2]],c='black',ls='--')
        #alpha = min(np.abs((vox.properties['n'][1] - 1)/0.1),1)
        #rec = Rectangle((vox.centroid[0] - vox.dx/2.,vox.centroid[2] - vox.dz/2.), vox.dx, vox.dz,facecolor='grey',alpha=0.2)
        #ax.add_patch(rec)
        for key in vox.lineSegments.keys():
            p1 = vox.lineSegments[key].origin
            p2 = vox.lineSegments[key].eval(vox.lineSegments[key].sep)
            ax.plot([p1[0],p2[0]],[p1[2],p2[2]],ls='-')
    plt.show()
    return ax

def plotOctTreeYZ(octTree,ax=None):
    import pylab as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        ax = plt.subplot(111)
        ax.set_xlabel('y')
        ax.set_ylabel('z')
    voxels = getAllDecendants(octTree)
    for vox in voxels:
        #plot S plane (2)
        for edge in vox.boundingPlanes[1].edges:
            p1 = edge.origin
            p2 = edge.eval(edge.sep)
            ax.plot([p1[1],p2[1]],[p1[2],p2[2]],c='black',ls='--')
        #alpha = min(np.abs((vox.properties['n'][1] - 1)/0.1),1)
        #rec = Rectangle((vox.centroid[0] - vox.dx/2.,vox.centroid[2] - vox.dz/2.), vox.dx, vox.dz,facecolor='grey',alpha=0.2)
        #ax.add_patch(rec)
        for key in vox.lineSegments.keys():
            p1 = vox.lineSegments[key].origin
            p2 = vox.lineSegments[key].eval(vox.lineSegments[key].sep)
            ax.plot([p1[1],p2[1]],[p1[2],p2[2]],ls='-')
    plt.show()
    return ax

def plotOctTree3D(octTree,model=None,rays=False):
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    voxels = getAllDecendants(octTree)
    vmax = np.max(model)
    vmin = np.min(model)
    i = 0
    for vox in voxels:
        if model is not None:
            p = ax.scatter(*vox.centroid,edgecolor=None,depthshade=True,c=model[i],norm=colors.Normalize(vmin = vmin,vmax = vmax))
            i += 1
        if rays:
            for key in vox.lineSegments.keys():
                p1 = vox.lineSegments[key].origin
                p2 = vox.lineSegments[key].eval(vox.lineSegments[key].sep)
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],ls='-')
    fig.colorbar(p)
    plt.show()


def testOtherSide():
    '''Return the idx of sibling in parent that is on other side of this plane.
        return -1 if otherside is parent boundary
        
        Make eight voxels to partition this voxel. 8x longer per layer
* 3 bits (a,b,c)
*           Top
*        +-----+-----+
*        ^  6  |  7  |
*        y N.W.| N.E.|
*        +-----O-x>--+
*        | S.W.| S.E.|
*        |  4  |  5  |
*        +-----+-----+
*           Bottom
*        +-----+-----+
*        ^  2  |  3  |
*        y N.W.| N.E.|
*        +-----O-x>--+
*        | S.W.| S.E.|
*        |  0  |  1  |
*        +-----+-----+
* 3 qubits (a1,a2),(b1,b2),(c1,c2)
*        +-----+
*        ^  3  |
*        z  N. |
*  +--<z-+--x>-#--z>-+-<x--+
*  |  W. ^  D. | E.  | U.  |
*  |  0  y  4  |  1  |  5  |
*  +-----O--x>-+-----+-----+
*        z  S. |
*        v  2  |
*        +-----+
    '''
    o = subDivide(OctTree([0,0,0.5],dx=1,dy=1,dz=100))
    #test BSE
    vox = o.children[1]
    #down
    print(getOtherSide(vox,4),-4-1)
    #south
    print(getOtherSide(vox,2),-2-1)
    #down
    print(getOtherSide(vox,1),-1-1)
    print(getOtherSide(vox,0),0)
    print(getOtherSide(vox,3),3)
    print(getOtherSide(vox,5),5)
    #test BSE
    vox = o.children[6]
    #down
    print(getOtherSide(vox,5),-5-1)
    #south
    print(getOtherSide(vox,3),-3-1)
    #down
    print(getOtherSide(vox,0),0-1)
    print(getOtherSide(vox,1),7)
    print(getOtherSide(vox,2),4)
    print(getOtherSide(vox,4),2)
    
if __name__ == '__main__':    
    #testOtherSide()
    try:
        octTree = loadOctTree('octTree_5levels.npy')
    except:
        octTree = OctTree([0,0,0.5],dx=10,dy=10,dz=10)
        subDivideToDepth(octTree,5)
        saveOctTree('octTree_5levels.npy',octTree)
    for i in range(20):
        ray = Ray(np.array([0,0,-epsFloat]),np.random.uniform(low=0,high=1,size=3),id=i)
        forwardRay(ray,octTree)
    plotOctTreeYZ(octTree,ax=None)
    


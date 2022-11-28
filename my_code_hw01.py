#-- my_code_hw01.py
#-- hw01 GEO1015.2022
#-- [Malte Schade]
#-- [5850282] 

import scipy
import numpy as np

def nn_xy(dt, kd, all_z, x, y): # Execution time: 0.2s
    """
    Function that interpolates with nearest neighbour method.
     
    Input:
        dt:     the DT of the input points (a startinpy object)
        kd:     the kd-tree of the input points 
        all_z:  an array with all the z values, same order as kd.data
        x:      x-coordinate of the interpolation location
        y:      y-coordinate of the interpolation location
    Output:
        z: the estimation of the height value, 
           (raise Exception if outside convex hull)
    """
    if not dt.is_inside_convex_hull(x, y):
        raise Exception("Outside convex hull")
    
    z = all_z[kd.query([x,y], workers=-1)[1]]
    return z


def idw_xy(dt, kd, all_z, x, y, power, radius): # Execution time: 1.4s
    """
    Function that interpolates with IDW
     
    Input:
        dt:     the DT of the input points (a startinpy object)
        kd:     the kd-tree of the input points 
        all_z:  an array with all the z values, same order as kd.data
        x:      x-coordinate of the interpolation location
        y:      y-coordinate of the interpolation location
        power:  power to use for IDW
        radius: search radius
¨    Output:
        z: the estimation of the height value, 
           (raise Exception if (1) outside convex hull or (2) no point in search radius
    """
    if not dt.is_inside_convex_hull(x, y):
        raise Exception("Outside convex hull")
    
    n = kd.query_ball_point([x,y], radius, workers=-1) 
    
    if not n:
        raise Exception("No point in search radius")
    
    d,i = kd.query([x,y], len(n), workers=-1)
    idw = np.power(d, -power)
    z = np.sum(np.multiply(idw, all_z[i]))/np.sum(idw)
    return z


def tin_xy(dt, kd, all_z, x, y): # Execution time: 4s
    """
    Function that interpolates linearly in a TIN.
     
    Input:
        dt:     the DT of the input points (a startinpy object)
        kd:     the kd-tree of the input points 
        all_z:  an array with all the z values, same order as kd.data
        x:      x-coordinate of the interpolation location
        y:      y-coordinate of the interpolation location
    Output:
        z: the estimation of the height value, 
           (raise Exception if outside convex hull)
    """
    if not dt.is_inside_convex_hull(x, y):
        raise Exception("Outside convex hull")
    
    p = dt.points[dt.locate(x, y)]
    l1 = ((p[1,1]-p[2,1])*(x-p[2,0])        + (p[2,0]-p[1,0])*(y-p[2,1]))\
       / ((p[1,1]-p[2,1])*(p[0,0]-p[2,0])   + (p[2,0]-p[1,0])*(p[0,1]-p[2,1]))
    l2 = ((p[2,1]-p[0,1])*(x-p[2,0])        + (p[0,0]-p[2,0])*(y-p[2,1]))\
       / ((p[1,1]-p[2,1])*(p[0,0]-p[2,0])   + (p[2,0]-p[1,0])*(p[0,1]-p[2,1]))
    l3 = 1-l1-l2
    
    z = (l1*p[0,2]+l2*p[1,2]+l3*p[2,2])
    return z


def nni_xy(dt, kd, all_z, x, y): # Execution time: 1240s
    """
    Function that interpolates with natural neighbour interpolation method (nni).
     
    Input:
        dt:     the DT of the input points (a startinpy object)
        kd:     the kd-tree of the input points 
        all_z:  an array with all the z values, same order as kd.data
        x:      x-coordinate of the interpolation location
        y:      y-coordinate of the interpolation location
    Output:
        z: the estimation of the height value, 
           (raise Exception if outside convex hull)
    """
    if not dt.is_inside_convex_hull(x, y):
        raise Exception("Outside convex hull")
    
    points = dt.points[1:,:2]
    A = scipy.spatial.Voronoi(points)
    B = scipy.spatial.Voronoi(np.append(points, [[x,y]], axis=0))
    b_ri = np.nonzero([np.isin(len(B.points)-1, rp) for rp in B.ridge_points])
    b_pi = [[p for p in row if p!=len(B.points)-1][0] for row in B.ridge_points[b_ri]]
    b_cc = [B.vertices[B.ridge_vertices[ri]] for ri in b_ri[0]]
    X_A = scipy.spatial.ConvexHull(B.vertices[B.regions[B.point_region[-1]]]).volume
    
    C_A = []
    for i in range(len(b_pi)):
        p_vi = np.array(A.regions[A.point_region[b_pi[i]]])
        p_vi = p_vi[np.nonzero(p_vi!=-1)]
        p_vy = A.vertices[p_vi]
        d_yy = [scipy.spatial.distance.pdist([A.points[b_pi[i]], p])[0] for p in p_vy]
        d_yx = [scipy.spatial.distance.pdist([[x,y], p])[0] for p in p_vy]
        C_1 = p_vy[np.nonzero(np.less_equal(d_yx, d_yy))[0]]
        C_2 = b_cc[i]
        C_A.append(scipy.spatial.ConvexHull(np.concatenate([C_1, C_2], axis=0)).volume)

    z = np.sum(np.multiply(np.divide(C_A, X_A), all_z[b_pi]))
    return z
    



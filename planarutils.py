#!/usr/bin/env python3
#
#   planarutils.py
#
#   Provide several utilities for planar geometry.  In particular:
#
#     PointNearPoint(dist, pointA, pointB)
#     PointNearSegment(dist, point, segment)
#     PointInTriangle(point, triangle)
#     PointNearTriangle(dist, point, triangle)
#
#     SegmentCrossSegment(segmentA, segmentB)
#     SegmentNearSegment(dist, segmentA, segmentB)
#     SegmentCrossTriangle(segment, triangle)
#
#   where
#     distance d                is a distance of proximity
#     point    p = (x,y)        is a point at x/y coordinates
#     segment  s = (p1, p2)     is the line segment between the points
#     triangle t = (p1, p2, p3) is the triangle with given corners
#
#   All functions return True or False.
#
import numpy as np


#
#   Proximity of Point to Point
#
def PointNearPoint(d, pA, pB):
    return ((pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 <= d**2)


#
#   Proximity of Point to Segment
#
def PointNearSegment(d, p, s):
    # Precompute the relative vectors.
    (vx, vy) = (s[1][0]-s[0][0], s[1][1]-s[0][1])
    (rx, ry) = (   p[0]-s[0][0],    p[1]-s[0][1])

    # Precompute the vector products.
    rTv = rx*vx + ry*vy
    rTr = rx**2 + ry**2
    vTv = vx**2 + vy**2

    # Check the point-to-point distance when outside the segment range.
    if (rTv <= 0):
        return (rTr <= d**2)
    if (rTv >= vTv):
        return (rTr - 2*rTv + vTv <= d**2)

    # Check the orthogonal point-to-line distance inside the segment range.
    return ((rx*vy - ry*vx)**2 <= vTv * d**2)


#
#   Segment Crossing Segment
#
def SegmentCrossSegment(sA, sB):
    # Precompute the relative vectors.
    (ux, uy) = (sA[1][0]-sA[0][0], sA[1][1]-sA[0][1])
    (vx, vy) = (sB[1][0]-sB[0][0], sB[1][1]-sB[0][1])
    (rx, ry) = (sB[0][0]-sA[0][0], sB[0][1]-sA[0][1])

    # Precompute the vector cross products.
    uXv = ux*vy - uy*vx
    rXu = rx*uy - ry*ux
    rXv = rx*vy - ry*vx

    # Check the intersection.
    if (uXv > 0):
        return ((rXu > 0) and (rXu < uXv) and (rXv > 0) and (rXv < uXv))
    else:
        return ((rXu < 0) and (rXu > uXv) and (rXv < 0) and (rXv > uXv))


#
#   Proximity of Segment to Segment
#
def EndpointsNearSegmentInterior(d, sA, sB):
    # Precompute the relative vectors.
    ( vx,  vy) = (sB[1][0]-sB[0][0], sB[1][1]-sB[0][1])
    (r1x, r1y) = (sA[0][0]-sB[0][0], sA[0][1]-sB[0][1])
    (r2x, r2y) = (sA[1][0]-sB[0][0], sA[1][1]-sB[0][1])

    # Precompute the vector cross products (orthogonal distances).
    r1Xv = r1x*vy - r1y*vx
    r2Xv = r2x*vy - r2y*vx

    # If the endpoints are on opposite sides of the line, this test
    # becomes irrelevant.  Skip and report.
    if (r1Xv * r2Xv <= 0):
        return (True, False)

    # Finaly check the only closer endpoint sA[0]/sA[1] vs the segment
    # sB[0]-sB[1].  The farther endpoint can never be the shortest distance.
    vTv = vx**2 + vy**2
    if abs(r2Xv) < abs(r1Xv):
        r2Tv = r2x*vx + r2y*vy
        return (False, ((r2Tv    >=   0) and
                        (r2Tv    <= vTv) and
                        (r2Xv**2 <= vTv * d**2)))
    else:
        r1Tv = r1x*vx + r1y*vy
        return (False, ((r1Tv    >=   0) and
                        (r1Tv    <= vTv) and
                        (r1Xv**2 <= vTv * d**2)))

def SegmentNearSegment(d, sA, sB):
    # First check the endpoints-to-endpoints.
    if (PointNearPoint(d, sA[0], sB[0]) or
        PointNearPoint(d, sA[0], sB[1]) or
        PointNearPoint(d, sA[1], sB[0]) or
        PointNearPoint(d, sA[1], sB[1])):
        return True

    # Then check the endpoints to segment interiors.  This also
    # reports whether the endpoints are on opposites sides of the
    # segement.  
    cross1, near = EndpointsNearSegmentInterior(d, sA, sB)
    if near:
        return True
    cross2, near = EndpointsNearSegmentInterior(d, sB, sA)
    if near:
        return True

    # If both crossed, the segments are intersecting.
    return (cross1 and cross2)


#
#   Proximity of Point to Triangle
#
def PointInTriangle(p, t):
    # Precompute the relative vectors.
    (apx, apy) = (t[0][0]-p[0], t[0][1]-p[1])
    (bpx, bpy) = (t[1][0]-p[0], t[1][1]-p[1])
    (cpx, cpy) = (t[2][0]-p[0], t[2][1]-p[1])

    # Precompute the cross products
    aXb = apx * bpy - apy * bpx
    bXc = bpx * cpy - bpy * cpx
    cXa = cpx * apy - cpy * apx
    
    # Determine whether the triangle is given CW (axb + bxc + cxa > 0) or CCW.
    if (aXb + bXc + cXa > 0):
        return (aXb >= 0) and (bXc >= 0) and (cXa >= 0)
    else:
        return (aXb <= 0) and (bXc <= 0) and (cXa <= 0)
    
def PointNearTriangle(d, p, t):
    # If inside, the point is considered "near".
    if PointInTriangle(p, t):
        return True

    # Check proximity to the three sides.  There is definitely
    # redundancy and hence inefficiency in these tests.
    return (PointNearSegment(d, p, (t[0], t[1])) or
            PointNearSegment(d, p, (t[1], t[2])) or
            PointNearSegment(d, p, (t[2], t[0])))


#
#   Intersection of Segment and Triangle
#
def SegmentCrossTriangle(s, t):
    return (PointInTriangle(s[0], t) or
            PointInTriangle(s[1], t) or
            SegmentCrossSegment(s, (t[0], t[1])) or
            SegmentCrossSegment(s, (t[1], t[2])) or
            SegmentCrossSegment(s, (t[2], t[0])))

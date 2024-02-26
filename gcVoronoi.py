#module for geo-constrained Voronoi tessellation, to spit out definite polygons that can be added to traces

from scipy.spatial import Voronoi
from shapely import geometry
import numpy as np

def points_to_coords(points_list):
    return list([p.coords[0] for p in points_list])

class gcVoronoi(Voronoi):
    def __init__(self, points_list: list, qhull_options: str | None = None) -> None:
        '''
        points_list is a list of shapely.geometry.Point, each representing a seed. this 
        is easy to get if importing e.g. a shapefile. make sure it's a list!
        '''
        self.seed_points = points_list
        coords_list = points_to_coords(points_list)
        super().__init__(coords_list, False, False, qhull_options)

        self._inf_segs = self._get_inf_segs()
        self._finite_regions, self._infinite_regions = self._polygonisation()
        self.unconstrained_polygons = self._collect_and_sort()
        self.constrained_polygons = None

    def _get_inf_segs(self, **kw):
        '''
        scipy.spatial.voronoi_plot_2d has an inbuilt method that takes line segments that
        would be infinite and approximates a terminus. This is necessary for the method's 
        plot output, but isn't used to support the voronoi itself. by incorporating this
        method as part of the Voronoi object and removing all plotting stuff, this method
        is now used to support the gcVoronoi polygon creation.
        so in case it's not clear this is an adaptation of scipy.spatial.voronoi_plot_2d
        '''
        if self.points.shape[1] != 2:
            raise ValueError("Voronoi diagram is not 2-D")

        center = self.points.mean(axis=0)
        ptp_bound = self.points.ptp(axis=0)

        finite_segments = []
        infinite_segments = []
        for pointidx, simplex in zip(self.ridge_points, self.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(self.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = self.points[pointidx[1]] - self.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = self.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if (self.furthest_site):
                    direction = -direction
                far_point = self.vertices[i] + direction * ptp_bound.max()

                infinite_segments.append([self.vertices[i], far_point])

        return infinite_segments
    
    def _polygonisation_prep(self):
        '''
        the points that are represented in infinite segments need to be drawn
        out for their polygonisation. this is messy and confusing
        infinite_segments = [[(x_coord of a vertex, y_coord of the same vertex),
          (x_coord of the created terminus, y_coord of the created terminus)], ...]

        firsts = coordinates of all vertices
        firsts_bare = x-coordinates of all vertices
        seconds = coordinates of all temini
        seconds_points = shapely points of all termini

        this prep is needed because ...

        '''
        firsts = [self._inf_segs[i][0] for i in range(len(self._inf_segs))]
        firsts_bare = [firsts[i][0] for i in range(len(firsts))]
        seconds = [self._inf_segs[i][1] for i in range(len(self._inf_segs))]
        seconds_points = [geometry.Point(seconds[i]) for i in range(len(seconds))]
        return firsts, firsts_bare, seconds, seconds_points

    def _polygonisation(self):
        '''
        Polygonisation method for finite cells is different to that for infinite cells
        cells (regions) are reported from voronoi object as lists of ids of vertices 
        that make up said regions. for a finite, this might be [55, 40, 39, 53],
        where each number is the index of the vertex in the list of vertices reported
        from the voronoi object. for infinite cells, there is one (or more) vertex
        that is infinity, therefore not reported in the vertex list from voronoi objects,
        meaning they have no id number, instead reported as -1. so an infinte region would
        look like [35, -1, 4, 34].

        '''
        #setup
        firsts, firsts_bare, seconds, seconds_points = self._polygonisation_prep()

        infinite_regions_reported = []
        regions_spatial = [] # for collecting polygons
        infinite_regions_spatial = [] # for collecting polygons

        #initial sort and finite region polygonisation
        for i in self.regions:
            if -1 in i:
                infinite_regions_reported.append(i)
            else:
                regions_spatial.append(geometry.Polygon(shell=[self.vertices[j] for j in i]))

        #infinite region polygonisation
        for i in range(len(infinite_regions_reported)):
            verts = infinite_regions_reported[i]
            inf_coords = []
            for j in range(len(verts)):
                item = verts[j]
                if item != -1:
                    inf_coords.append(self.vertices[item])
                else:
                    inf_coords.append(seconds[firsts_bare.index(self.vertices[verts[j-1]][0])]) #this is a search of our collection of termini points ("seconds_points") - they're only searchable by their corresponding vertex, i.e "first"
                    try:
                        inf_coords.append(seconds[firsts_bare.index(self.vertices[verts[j+1]][0])]) # throws error if -1 is last vertex
                    except:
                        inf_coords.append(seconds[firsts_bare.index(self.vertices[verts[0]][0])]) # takes advantage of cycle of vertices - still moving forward
            poly = geometry.Polygon(shell=inf_coords)
            infinite_regions_spatial.append(poly)

        return regions_spatial, infinite_regions_spatial
    
    def _collect_and_sort(self):
        polygons = self._finite_regions + self._infinite_regions
        sorted_polygons = []
        for i in self.seed_points:
            for j in polygons:
                if i.within(j):
                    sorted_polygons.append(j)
                    polygons.remove(j)
                    break
        return sorted_polygons
    
    def geo_constrain(self, boundary_poly: geometry.Polygon):
        '''
        use to constrain the polygons to a geometric border, which helps for
        clean plotting or interfacing with other data, e.g. WorldPop
        
        please ensure that boundary_poly is, in fact, a geometry.Polygon. if
        you are using a shapefile of a country's administrative border and
        importing as a geoPandas df, you might be inclined to put df.geometry
        as boundary_poly here. but df.geometry is a column/series, not the
        polygon within it. use df.geometry[0] instead.
        
        '''

        constrained_polygons = []
        for i in self.unconstrained_polygons:
            constrained_polygons.append(i.intersection(boundary_poly))
        

        self.constrained_polygons = constrained_polygons
        return constrained_polygons
    
    def to_geoSeries(self, constrained=True):
        from geopandas import GeoSeries
        if  constrained:
            if self.constrained_polygons:
                return GeoSeries(self.constrained_polygons)
            else:
                print('error: polygons have not been constrained yet. please apply .geo_constrain to the Voronoi object with a suitable boundary')
        else:
            return GeoSeries(self.unconstrained_polygons)


if __name__ == '__main__':
    import geopandas as gpd
    gdf = gpd.read_file('preschools.shp')
    vor = gcVoronoi(gdf.geometry.to_list())
    border = gpd.read_file('uppsala.shp')
    # print(vor.unconstrained_polygons)
    vor.geo_constrain(border.geometry[0])
    # print(vor.constrained_polygons)

    import matplotlib.pyplot as plt
    vor.to_geoSeries().plot()
    plt.show()

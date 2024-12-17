# this file contains all functions that can be used together for the construction of constrained voronoi tessellations on certain countries and the populations within them
# section 1 = the pure Voronoi functionality
# section 2 = getting worldpop data from the API
# section 3 = the main, holistic functions to draw voronoi tessellations from a shapefile, interface with worldpop to get the pop within each tessellation, and render an image as output

# SUBJECT TO LICENCE - ATTRIBUTE NEIL MAJITHIA




from scipy.spatial import Voronoi
from shapely import geometry
import numpy as np
import requests
import geopandas as gpd
from tqdm import tqdm
from rasterio.io import MemoryFile
import pygadm

##################### section 1

def points_to_coords(points_list):
    '''
    Helper function that turns a list of shapely.geometry.Point into a list of respective coordinates
    '''
    return list([p.coords[0] for p in points_list])

class gcVoronoi(Voronoi):
    '''
    Object class, inheriting from the scipy.spatial.Voronoi object, extended to have the functionality necessary to draw a
    Voronoi tessellation that is non-infinite. This is important as the original scipy Voronoi methods do not return useful
    shapely.geometry.Polygon objects, purely because of the fact that some Voronoi regions may be infinite. 

    Initialise with a list of shapely.geometry.Point objects that you want to draw a Voronoi tessellation from.

    '''
    def __init__(self, points_list: list, qhull_options: str | None = None) -> None:
        '''
        points_list is a list of shapely.geometry.Point, each representing a seed. this 
        is easy to get if importing e.g. a shapefile. make sure it's a list!
        '''
        self.seed_points = points_list
        coords_list = points_to_coords(points_list)
        super().__init__(coords_list, False, False, qhull_options) # draw the scipy.spatial.Voronoi tessellation and get data outputs

        self._inf_segs = self._get_inf_segs()
        self._finite_regions, self._infinite_regions = self._polygonisation()
        self.unconstrained_polygons = self._collect_and_sort()
        self.constrained_polygons = None

    def _get_inf_segs(self, **kw):
        '''
        The aim of this entire object is to make the potentially infinite voronoi regions
        into definitely finite geometry.Polygons.

        scipy.spatial.voronoi_plot_2d has an inbuilt method that takes line segments of voronoi regions 
        that would be infinite and approximates a terminus. This is necessary for the method's 
        plot output, but isn't used to support the voronoi itself. By incorporating this
        method as part of the Voronoi object and removing all plotting components, this method
        is now used to turn infinite regions finite and therefore support the gcVoronoi polygon.

        In actuality, this method returns the line segments which make up the infinite regions. Each
        subset of these corresponds to one region, but importantly, no subset works as a geometry.Polygon by
        itself: the line segments generated in this method are always one line short from actually forming 
        a polygon. As a result, the results of this method need to be passed to a polygonisation method.

        In case it's not clear this is an adaptation of scipy.spatial.voronoi_plot_2d
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
        The points that are represented in the line segments coming out of _get_inf_segs need to be drawn
        out for their polygonisation. this is messy and confusing. 
        infinite_segments = [[(x_coord of a vertex, y_coord of the same vertex),
          (x_coord of the created terminus, y_coord of the created terminus)], ...]
        within this method:
        firsts = coordinates of all starting vertices of the line segments
        firsts_bare = x-coordinates of all starting vertices of the line segments
        seconds = coordinates of all temini of the line segments
        seconds_points = shapely points of all termini of the line segments

        this prep is needed because it feeds into a relatively simple nested loop setup in the polygonisation
        method

        '''
        firsts = [self._inf_segs[i][0] for i in range(len(self._inf_segs))]
        firsts_bare = [firsts[i][0] for i in range(len(firsts))]
        seconds = [self._inf_segs[i][1] for i in range(len(self._inf_segs))]
        seconds_points = [geometry.Point(seconds[i]) for i in range(len(seconds))]
        return firsts, firsts_bare, seconds, seconds_points

    def _polygonisation(self):
        '''
        The polygonisation method for finite cells is different to that for infinite cells.
        Cells (regions) are reported from original scipy Voronoi object as lists of ids of vertices 
        that make up said regions. for a finite, this might be [55, 40, 39, 53],
        where each number is the index of the vertex in the list of vertices reported
        from the voronoi object. for infinite cells, there is one (or more) vertex
        that is infinity, therefore not reported in the vertex list from voronoi objects,
        meaning they have no id number, instead reported as -1. so an infinte region would
        look like [35, -1, 4, 34].

        The polygonisation method turns these lists into polygons. For infinite regions,
        the _prep method used in the beginning works to allow -1s to be filled in with a
        smart nested loop system.

        This method returns lists of shapely.spatial.geometry.Polygon, one list for non-finite
        regions and one for would-be-infinite-but-now-finite regions (haha). each polygon represents
        a Voronoi region on the tessellation.

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
        '''
        Aggregate polygons and sort them to be in the same sequence as seeds.
        Returns one list of sorted_polygons.
        '''

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
        Uses to constrain the polygons to a geometric border, which helps for
        clean plotting or interfacing with other data, e.g. WorldPop
        
        please ensure that boundary_poly is, in fact, a geometry.Polygon. if
        you are using a shapefile of a country's administrative border and
        importing as a geoPandas df, you might be inclined to put df.geometry
        as boundary_poly here. but df.geometry is a column/series, not the
        polygon within it. use df.geometry[0] instead.

        returns a list of constrained polyogons.
        
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

##################### section 2


def tif_dataset_to_csv(dataset, nodata=-99999.0):

    '''
    When data is received from the WorldPop API, it comes as a .tif.
    .tif files are unique geospatial files that encode data as an image,
    using 'bands' of the image to encode different layers of data and
    using kwargs in the metadata to instruct users on how to turn the data
    coordinate-based. 

    This method works to do this transformation for us. WorldPop uses
    -99999.0 as their signal for no data being at a coordinate.
    
    '''


    population = dataset.read(1)  # Read the first band
    transform = dataset.transform
    width = dataset.width
    height = dataset.height
    crs = dataset.crs

    arr = []

    # Iterate over each pixel
    print("Mapping pixels to coordinates...")
    for row in tqdm(range(height), desc='Rows processed'):
        for col in range(width):
            pop = population[row, col]
            if pop == nodata:
                continue  # Skip no-data values

            lat, lon = transform * (col+0.5, row+0.5)  # Center of the pixel

            data_point = [lat, lon, pop]
            arr.append(data_point)

    df = gpd.GeoDataFrame(data=arr, columns=['X', 'Y', 'Z'], crs=crs, geometry=[geometry.Point(i[0], i[1]) for i in arr])
    return df, crs


def get_worldpop_population_data(iso3='SEN', return_crs = False):
    '''
    This method queries the WorldPop API for 'The spatial distribution of population in
    2020 with country total adjusted to match the corresponding UNPD estimate' in a 
    chosen country, with this being done with an iso3 code. The default is Senegal.

    The method returns a geopandas dataframe and, optionally, the crs of the data.
    '''
   
    request_for_filepath = requests.get('https://www.worldpop.org/rest/data/pop/wpicuadj1km', params={'iso3': iso3})
    if request_for_filepath.status_code == 200:
        filepath = request_for_filepath.json()['data'][-1]['files'][0]
        if filepath.endswith('tif'):
            pass
        else:
            filepath = request_for_filepath.json()['data'][-1]['files'][1]
    else:
        raise Exception(f'Failed to request Worldpop data. Check iso3 input. Status code: {request_for_filepath.status_code}')

    # Download the file content
    response = requests.get(filepath)
    if response.status_code == 200:
        file_content = response.content
    else:
        raise Exception(f'Failed to download file. Status code: {response.status_code}')

    # Use MemoryFile to open the .tif from bytes
    with MemoryFile(file_content) as memfile:
        with memfile.open() as dataset:
            df, crs = tif_dataset_to_csv(dataset)
    if return_crs:
        return df, crs
    else:
        return df

def get_admin_border_geom_lvl0(iso3='SEN', return_bbox=False):
    '''
    this method relies on pygadm to get the level 0 administrative border 
    for a chosen country, using iso3 code. Optionally can also return the 
    bounding box of the country's regions, which is useful for plotting.
    The bounding box is buffered by 0.5km.  
    '''
    gdf = pygadm.Items(admin=iso3, content_level=0)
    if return_bbox:
        bbox = gdf['geometry'].values[0].buffer(distance=0.5).bounds
        return gdf[['geometry', 'NAME_0']], bbox
    else:
        return gdf[['geometry', 'NAME_0']]
        

##################### section 3

def draw_constrained_voronoi(shapefile_filepath, boundary_poly, return_hosps=False):
    '''
    Takes a healthsites.io shapefile and a boundary polygon and returns the constrained
    voronoi tessellation these correspond to. make sure that the shapefile country and the 
    boundary_poly country are the same!

    return_hosps is an optional parameter - if true, this method returns a geodataframe of
    the shapefile.
    '''

    gdf = gpd.read_file(shapefile_filepath)
    hosps = gdf[(gdf['amenity'] == 'hospital' )|( gdf['amenity'] == 'clinic')].drop_duplicates(subset='geometry', keep='first')

    vor = gcVoronoi(hosps.geometry.to_list())
    vor.geo_constrain(boundary_poly)
    con_gS = vor.to_geoSeries(constrained=True)

    if return_hosps:
        return con_gS, hosps
    else:
        return con_gS


def interface_w_worldpop(constrained_voronoi_gs, worldpop_df, osm_ids, crs=None):
    '''
    This method layers the constrained voronoi tessellation with a worldpop data frame,
    using osm_ids of hospitals as their unique identifiers.

    There is a lot of room to optimise this. It takes roughly 6 minutes for Senegal.
    I've used tqdm so that you have a loading bar and can monitor progress and ETA.

    Returns the layered voronoi tessellation as a geopandas DataFrame, with the 
    hospital osm_id in one column, the population in its region in another,
    and the voronoi tessellation regions presented in the geometry column. 
    '''
    pop_list = np.zeros(len(constrained_voronoi_gs))
    print('Interfacing with Worldpop...')
    for i in tqdm(worldpop_df.index, desc='Points processed'):
        # print('point: ', i, end='\r')
        for j in range(len(constrained_voronoi_gs)):
            if worldpop_df.loc[i, 'geometry'].within(constrained_voronoi_gs[j]):
                pop_list[j] += worldpop_df.loc[i, 'Z']
                break
    
    output = gpd.GeoDataFrame(data={
        # 'geometry': constrained_voronoi_gs,
        'pop': pop_list,
        'hosp_osm_id': osm_ids.reset_index()['osm_id']
    }, geometry=constrained_voronoi_gs, crs=crs)

    return output

def render_static(voronoi_gdf, hosps_gdf, boundary_gdf, bbox=None, country_name='Senegal', **kwargs):
    '''
    With all outputs, rendering a static matplotlib plot.
    '''


    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import contextily as cx

    legend_elements = [
    Line2D([0], [0], color='w', marker='P', markerfacecolor='green', markersize=15, label='Healthsite with confirmed emergency room'),
    Line2D([0], [0], color='w', marker='P', markerfacecolor='black', markersize=15, label='Healthsite with no confirmed emergency room')
    ]

    fig, ax = plt.subplots(figsize=(7, 9), dpi=100)

    boundary_gdf.plot(ax=ax, color='none')
    voronoi_gdf.plot(ax=ax, column='pop', cmap='OrRd', legend=True, 
                legend_kwds={'label': 'Population',
                            'orientation': 'horizontal',
                            'shrink': 0.8,
                            'aspect': 25,
                            'pad': 0.01,
                            'fraction': 0.1, 
                            },
                edgecolor='grey', linewidth=0.5, alpha=0.7)

    hosps_gdf[hosps_gdf['emergency'] == 'yes'].plot(ax=ax, color='green', marker='P', markersize=50, alpha=0.7)
    hosps_gdf[hosps_gdf['emergency'] != 'yes'].plot(ax=ax, color='black', marker='P', markersize=50, alpha=0.7)

    if bbox:
        ax.set_xlim(left=bbox[0]-0.5, right=bbox[2]-0.5)
        ax.set_ylim(bottom=bbox[1], top=bbox[3]-0.5)




    ax.set_title(f'Voronoi Diagram: Hospitals and Clinics Service Areas in {country_name}', fontsize=16, pad=20)
    ax.set_axis_off()


    cx.add_basemap(ax, crs=hosps_gdf.crs, source=cx.providers.CartoDB.Positron, zoom=9)

    ax.annotate('N', xy=(0.95, 0.94), xycoords='axes fraction', fontsize=12, ha='center', va='center')
    ax.annotate('↑', xy=(0.95, 0.98), xycoords='axes fraction', fontsize=20, ha='center', va='center')

    ax.legend(handles=legend_elements, loc='lower right', shadow=True, fontsize='small')

    fig.text(
    0.5,              # x-coordinate (centered)
    0.01,             # y-coordinate (bottom margin)
    "Data: healthsites.io (2024), WorldPop (2020) | Drawn with NevadaM/gcVoronoi", 
    ha="center",      # Horizontal alignment
    fontsize=8,       # Font size
    color="gray"      # Text color
    )


    plt.tight_layout()
    plt.show()

    return fig, ax


def render_interactive(voronoi_gdf, hosps_gdf, boundary_gdf, bbox, country_name='Senegal', **kwargs):
    '''
    With all outputs, rendering an interactive Folium map 
    '''

    import folium
    import folium.plugins as plugins
    x, y = geometry.LineString([[bbox[0], bbox[1]], [bbox[2], bbox[3]]]).centroid.xy
    m = folium.Map(location=(y[0], x[0]), tiles='OpenStreetMap', zoom_start=6)
    
    voronoi_gdf.explore(
        m = m,
        column='pop',
        # scheme='quantiles',
        cmap='OrRd',
        legend=True,
        legend_kwds={
            'label': 'Population',
            'orientation': 'horizontal'
        },
        tooltip=['pop', 'hosp_osm_id'],
        popup=True,
        # tiles='CartoDB positron',
        name='Population'  
    )

    boundary_gdf.explore(m=m, color='none', name='Boundary')


    hosps_gdf[hosps_gdf['emergency'] == 'yes'].explore(
        m=m,
        color='green',
        # marker_type='circle',
        # radius=5,
        tooltip=['name', 'emergency'],
        name='Hospitals with Emergency Rooms'
    )

    hosps_gdf[hosps_gdf['emergency'] != 'yes'].explore(
        m=m,
        color='red',
        # marker_type='circle',
        # radius=5,
        tooltip=['name', 'emergency'],
        name='Hospitals without Emergency Rooms'
    )

    title_html = f'''
             <h3 align="center" style="font-size:16px">
             <b>Voronoi Diagram: Hospital and Clinic Service Areas in {country_name}</b>
             </h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))

    plugins.MeasureControl(position='bottomleft').add_to(m)
    plugins.Fullscreen().add_to(m)
    folium.LayerControl().add_to(m)

    m.fit_bounds(bounds=bbox)
    return m




def DEPRECATED_get_raw_healthsites(country='Senegal'):

    # this API is too difficult to work with. the 50 request daily limit conflicts with the annoying pagination of responses that isn't like true pagination
    # so let's just move on and pretend that somehow we have the shapefile we need for a country

    # # Initialize a client & load the schema document
    # client = coreapi.Client()
    # schema = client.get("https://healthsites.io/api/docs/")

    # results = []

    # # Interact with the API endpoint
    # monitor=100
    # page = 1
    # while monitor != 0:
    #     action = ["facilities", "list"]
    #     params = {
    #         "api-key": API_KEY,
    #         "page": page,
    #         "country": country,
    #         # "extent": ...,
    #         # "from": ...,
    #         # "to": ...,
    #         # "flat-properties": True,
    #         # "tag-format": ...,
    #         "output": 'json',
    #     }
    #     result = client.action(schema, action, params=params)
    #     results.append(result)


    #     monitor = len(result)
    #     page+=1

    # return results
    
    
    return


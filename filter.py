import os
import sqlite3

import folium
import planetary_computer
import rasterio
import requests
from lambert import Lambert93, convertToWGS84Deg
from pyproj import Transformer
from pystac_client import Client
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.wkt import loads


# Class to store temporality data
class temporality():
    def __init__(self):
        self.batiment_rid = None
        self.parcelle_rid = None
        self.geom_batiment = None
        self.geom_parcelle = None        

# Class to find changes between before and after databases
class changeFinder():
    """
    Attributes:
    _conn_b (sqlite3.Connection): Connection to the before database.
    _cursor_b (sqlite3.Cursor): Cursor for the before database.
    _conn_a (sqlite3.Connection): Connection to the after database.
    _cursor_a (sqlite3.Cursor): Cursor for the after database.
    Methods:
    __init__(db_path_b, db_path_a):
        Initializes the changeFinder with paths to the before and after databases.
    get_table():
        Retrieves relevant columns from the databases and returns temporality objects for both databases.
    close():
        Closes the database connections.
    find_changes():
        Finds changes between the two databases and returns merged regions of interest.
    filter_on_area(ROI):
        Sorts regions of interest based on the area of the building and keeps the top 20% of the biggest buildings.
    merge_on_parcelle(ROI):
        Merges regions of interest on the same parcelle and returns a dictionary of merged regions.
    get_b_area(region):
        Calculates and returns the area of a building.
    generate_map(ROI):
        Generates a map with the regions of interest and saves it as map.html.
    global_bound(ROI):
        Calculates and returns the global geometry bound of all regions of interest.
    get_global_bound(global_geom):
        Generates a map with the global geometry bound and saves it as global_map.html.
    """
    def __init__(self, db_path_b, db_path_a):
        # Connect to the before database
        self._conn_b = sqlite3.connect(db_path_b)
        self._conn_b.enable_load_extension(True)
        self._conn_b.load_extension("mod_spatialite")
        self._cursor_b = self._conn_b.cursor()

        # Connect to the after database
        self._conn_a = sqlite3.connect(db_path_a)
        self._conn_a.enable_load_extension(True)
        self._conn_a.load_extension("mod_spatialite")
        self._cursor_a = self._conn_a.cursor()

    def get_table(self):
        """
        Get relevant columns from the databases.
        Gets column object rid from tables geo_batiment and geo_parcelle.

        Returns:
        Both temporality objects
        """
        before, after = [], []
        # SQL query to get relevant columns
        query = "SELECT ST_AsText(b.geom) AS batiment_geom,ST_AsText(p.geom) AS parcelle_geom, b.object_rid as batiment_rid, p.object_rid as parcelle_rid FROM geo_batiment AS b JOIN geo_batiment_parcelle AS bp ON b.geo_batiment = bp.geo_batiment JOIN geo_parcelle AS p ON bp.geo_parcelle = p.geo_parcelle;"
        
        # Execute query on before database
        self._cursor_b.execute(query)
        rows_b = self._cursor_b.fetchall()
        
        # Execute query on after database
        self._cursor_a.execute(query)
        rows_a = self._cursor_a.fetchall()
        
        # Process rows from before database
        for row in rows_b:
            t = temporality()
            t.parcelle_rid = row[3]
            t.batiment_rid = row[2]
            t.geom_batiment = row[0]
            t.geom_parcelle = row[1]
            before.append(t)
        
        # Process rows from after database
        for row in rows_a:
            t = temporality()
            t.parcelle_rid = row[3]
            t.batiment_rid = row[2]
            t.geom_batiment = row[0]
            t.geom_parcelle = row[1]
            after.append(t)
        
        return before, after
    
    # Close database connections
    def close(self):
        self._conn_b.close()
        self._conn_a.close()
    
    def find_changes(self):
        """
        Find changes between two databases.
        """
        before, after = self.get_table()
        num_changes = 0
        ROI = []
        # Get list of batiment_rid from before and after databases
        before_rids = [t.batiment_rid for t in before]
        after_rids = [t.batiment_rid for t in after]
        
        # Count changes by comparing batiment_rid
        for a in after:
            if a.batiment_rid not in before_rids and a.geom_batiment is not None and a.geom_parcelle is not None:
                num_changes += 1
                ROI.append(a)
        # Close database connections
        self.close()
        
        # Print number of changes found
        print(f"Found {num_changes} changes between the two databases.")
        ROI_filtered = self.filter_on_area(ROI)

        return self.merge_on_parcelle(ROI_filtered)

    def filter_on_area(self, ROI):
        """
        Sort ROI on area of the building using a key (we don't have to store the areas in another array as such).
        We then keep the top 20% of the biggest buildings.
        """
        ROI.sort(key=lambda x: self.get_b_area(x), reverse=True)
        length = len(ROI)
        return ROI[:int(0.1*length)]
        
    def merge_on_parcelle(self, ROI):
        """
        Merge regions of interest on the same parcelle.
        """
        ROI_dict = {}
        for r in ROI:
            if r.parcelle_rid not in ROI_dict:
                ROI_dict[r.parcelle_rid] = [r]
            else:
                ROI_dict[r.parcelle_rid].append(r)
        return ROI_dict

    def get_b_area(self, region):
        """
        Get area of a building.
        """
        geom = loads(region.geom_batiment)
        return geom.area

    def generate_map(self, ROI):
        """
        Generate a map with the regions of interest and saves it as map.html.
        """
        centroid = loads(ROI[list(ROI.keys())[0]][0].geom_parcelle).centroid
        pt = convertToWGS84Deg(centroid.x, centroid.y, Lambert93)
        m = folium.Map(location=[pt.getY(), pt.getX()], zoom_start=14)
        for rd in ROI.values():
            for geom in [rd[0].geom_parcelle]:
                # Convert Shapely geometry to GeoJSON
                geojson = mapping(loads(geom))
                # Convert coordinates from Lambert 93 to WGS84
                for l, feature in enumerate(geojson['coordinates']):
                    new_feature = []
                    for i, polygon in enumerate(feature):
                        new_feature.append([])
                        for j, coord in enumerate(polygon):
                            pt = convertToWGS84Deg(coord[0], coord[1], Lambert93)
                            new_feature[i].append([pt.getX(), pt.getY()])
                    geojson['coordinates'][l] = new_feature

                # Add to map with styling
                folium.GeoJson(
                    geojson,
                    style_function=lambda x: {
                        'fillColor': 'green',
                        'color': 'green',
                        'weight': 2,
                        'fillOpacity': 0.5
                    }
                ).add_to(m)

            for r in rd:
                for geom in [r.geom_batiment]:
                    # Convert Shapely geometry to GeoJSON
                    geojson = mapping(loads(geom))
                    # Convert coordinates from Lambert 93 to WGS84
                    for l, feature in enumerate(geojson['coordinates']):
                        new_feature = []
                        for i, polygon in enumerate(feature):
                            new_feature.append([])
                            for j, coord in enumerate(polygon):
                                pt = convertToWGS84Deg(coord[0], coord[1], Lambert93)
                                new_feature[i].append([pt.getX(), pt.getY()])
                        geojson['coordinates'][l] = new_feature

                    # Add to map with styling
                    folium.GeoJson(
                        geojson,
                        style_function=lambda x: {
                            'fillColor': 'blue',
                            'color': 'blue',
                            'weight': 2,
                            'fillOpacity': 0.5
                        }
                    ).add_to(m)
                    
                    # Add bounding box with -50m and +50m in all directions
                    centroid = loads(geom).centroid
                    bbox = [
                        [centroid.y - 160, centroid.x - 160],
                        [centroid.y - 160, centroid.x + 160],
                        [centroid.y + 160, centroid.x + 160],
                        [centroid.y + 160, centroid.x - 160],
                        [centroid.y - 160, centroid.x - 160]
                    ]
                    bbox_wgs84 = [[convertToWGS84Deg(pt[1], pt[0], Lambert93).getY(), convertToWGS84Deg(pt[1], pt[0], Lambert93).getX()] for pt in bbox]
                    folium.PolyLine(bbox_wgs84, color="yellow", weight=2.5, opacity=1).add_to(m)
                    
        # Save the map to an HTML file and display
        output_file = "map.html"
        m.save(output_file)
        print(f"Map saved as {output_file}. Open this file in a browser to view the geometries.")

    def global_bound(self, ROI):
        global_geom = None
        for rd in ROI.values():
            for r in rd: 
                geom = loads(r.geom_batiment)
                global_geom = geom if global_geom is None else global_geom.union(geom)
        return global_geom

    def get_global_bound(self, global_geom):
        # Generate a map with the global geometry bound and save it as global_map.html
        centroid = global_geom.centroid
        pt = convertToWGS84Deg(centroid.x, centroid.y, Lambert93)
        m = folium.Map(location=[pt.getY(), pt.getX()], zoom_start=14)
        # Convert Shapely geometry to GeoJSON
        geojson = mapping(global_geom)
        # Convert coordinates from Lambert 93 to WGS84
        for l, feature in enumerate(geojson['coordinates']):
            new_feature = []
            for i, polygon in enumerate(feature):
                new_feature.append([])
                for j, coord in enumerate(polygon):
                    pt = convertToWGS84Deg(coord[0], coord[1], Lambert93)
                    new_feature[i].append([pt.getX(), pt.getY()])
            geojson['coordinates'][l] = new_feature

        # Add to map with styling
        houses = folium.GeoJson(
            geojson,
            style_function=lambda x: {
                'fillColor': 'red',
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.5
            }
        ).add_to(m)

        # Add bounding box with -50m and +50m in all directions
        bounds = global_geom.bounds
        bbox = [
            [bounds[1] - 100, bounds[0] - 100],
            [bounds[1] - 100, bounds[2] + 100],
            [bounds[3] + 100, bounds[2] + 100],
            [bounds[3] + 100, bounds[0] - 100],
            [bounds[1] - 100, bounds[0] - 100]
        ]
        bbox_wgs84 = [[convertToWGS84Deg(pt[1], pt[0], Lambert93).getY(), convertToWGS84Deg(pt[1], pt[0], Lambert93).getX()] for pt in bbox]
        line = folium.PolyLine(bbox_wgs84, color="yellow", weight=2.5, opacity=1).add_to(m)

        # Save the map to an HTML file and display
        output_file = "global_map.html"
        m.save(output_file)
        print(f"Global map saved as {output_file}. Open this file in a browser to view the global geometry.")
        return bbox_wgs84, line, houses, bbox

if __name__ == "__main__":
    # Paths to before and after databases
    db_path_b = "./palaiseau_2018.sqlite"
    db_path_a = "./palaiseau_2024.sqlite"
    
    # Create changeFinder object and find changes
    cf = changeFinder(db_path_b, db_path_a)
    ROI_dict = cf.find_changes()
    print(f"Keeping {len(ROI_dict)} regions of interest.")
    # Show geometries of the regions of interest on a map
    #cf.generate_map(ROI_dict)
    global_geom = cf.global_bound(ROI_dict)
    bbox, max_bound, houses,bbox_lambert = cf.get_global_bound(global_geom)


    # Reorganize bbox to swap each latitude and longitude
    bbox = [[pt[1], pt[0]] for pt in bbox]


    # Initialize the Copernicus Open Access Hub STAC API
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)    

    # Define the search criteria
    search_criteria = {
        "collections": ["sentinel-2-l2a"],
        "datetime": "2024-01-01/2024-12-31",
        "bbox": [bbox[0][0], bbox[0][1],bbox[2][0],bbox[2][1]],
        "query": {"eo:cloud_cover": {"lt": 10}}
    }

    # Search for Sentinel-2 data
    search = catalog.search(**search_criteria)
    items = search.items()

    item = next(items)

    asset_url = item.assets["visual"].href     


    output_path = os.path.join("./data/", f"{item.id}_visual.tif")
    if not os.path.exists(output_path):
        response = requests.get(asset_url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print(f"Downloaded {output_path}")
    else:
        print(f"{output_path} already exists. Skipping download.")

    #Convert bbox to EPSG:32631
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631")
    bbox_32631 = [transformer.transform(pt[1], pt[0]) for pt in bbox]  

    bbox_wkt = f"POLYGON(({bbox_32631[0][0]} {bbox_32631[0][1]},{bbox_32631[1][0]} {bbox_32631[1][1]},{bbox_32631[2][0]} {bbox_32631[2][1]},{bbox_32631[3][0]} {bbox_32631[3][1]},{bbox_32631[0][0]} {bbox_32631[0][1]}))"

    print("Bounding box in {}:".format(bbox_wkt))

    with rasterio.open(output_path) as src:
        #Print src coordinates
        print(src.bounds)
        out_image, out_transform = mask(src, [mapping(loads(bbox_wkt))], crop=True)
        out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    output_cropped_path = os.path.join("./data/", f"{item.id}_visual_cropped.tif")
    with rasterio.open(output_cropped_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Cropped image saved as {output_cropped_path}")

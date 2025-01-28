import os
import sqlite3
from datetime import datetime

import folium
import numpy as np
import planetary_computer
import rasterio
import rasterio.features
import requests
from lambert import Lambert93, convertToWGS84Deg
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
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
        self.creat_date = None

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
        query = "SELECT ST_AsText(b.geom) AS batiment_geom,ST_AsText(p.geom) AS parcelle_geom, b.object_rid as batiment_rid, p.object_rid as parcelle_rid, b.creat_date as creat_date FROM geo_batiment AS b JOIN geo_batiment_parcelle AS bp ON b.geo_batiment = bp.geo_batiment JOIN geo_parcelle AS p ON bp.geo_parcelle = p.geo_parcelle;"
        self.insee = self._cursor_a.execute("SELECT lot FROM geo_parcelle LIMIT 1;").fetchone()[0]
        y1 = self._cursor_b.execute("SELECT annee FROM geo_batiment LIMIT 1;").fetchone()[0]
        y2 = self._cursor_a.execute("SELECT annee FROM geo_batiment LIMIT 1;").fetchone()[0]
        self.years = [y1,y2]
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
            t.creat_date = datetime.strptime(row[4], "%Y-%m-%d")
            before.append(t)
        
        # Process rows from after database
        for row in rows_a:
            t = temporality()
            t.parcelle_rid = row[3]
            t.batiment_rid = row[2]
            t.geom_batiment = row[0]
            t.geom_parcelle = row[1]
            t.creat_date = datetime.strptime(row[4], "%Y-%m-%d")
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
        before_geom = [t.geom_batiment for t in before] 
        # transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        
        for a in after:
            #     # Convert Lambert93 to Lambert99 CC50 on the fly and count changes
            #     geojson = mapping(loads(a.geom_batiment))
            #     for l, feature in enumerate(geojson['coordinates']):
            #         new_feature = []
            #         for i, polygon in enumerate(feature):
            #             new_feature.append([])
            #             for j, coord in enumerate(polygon):
            #                 x, y = transformer.transform(coord[1], coord[0])
            #                 new_feature[i].append([x, y])
            #     geojson['coordinates'][l] = new_feature
            #     #Create MultiPolygon from geojson
            #     a.geom_batiment = to_wkt(from_geojson('{"type": "MultiPolygon", "coordinates": ' + str(geojson['coordinates']) + '}'))
            if (a.geom_batiment not in before_geom) and (a.geom_batiment is not None) and (a.geom_parcelle is not None) and (a.creat_date > datetime.strptime(self.years[0], "%Y")):
                num_changes += 1
                ROI.append(a)

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
        self.houses = houses
        return bbox_wgs84

    def get_cropped_map(self, ROI_dict,year):
        global_geom = self.global_bound(ROI_dict)
        bbox= self.get_global_bound(global_geom)

            # Reorganize bbox to swap each latitude and longitude
        bbox = [[pt[1], pt[0]] for pt in bbox]


        # Initialize the Copernicus Open Access Hub STAC API
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)    

        print(f"Searching for Sentinel-2 data in {year} ...")
        if year == "2017":
            year = "2016"
        # Define the search criteria
        search_criteria = {
            "collections": ["sentinel-2-l2a"],
            "datetime": f"{year}-01-01/{year}-12-31",
            "bbox": [bbox[0][0], bbox[0][1],bbox[2][0],bbox[2][1]],
            "query": {"eo:cloud_cover": {"lt": 10}}
        }

        # Search for Sentinel-2 data
        search = catalog.search(**search_criteria)
        items = search.items()

        item = next(items)

        # Define band URLs
        band_urls = {
            "red": item.assets["B04"].href,
            "green": item.assets["B03"].href,
            "blue": item.assets["B02"].href,
            "nir": item.assets["B08"].href,
            "swir1": item.assets["B11"].href
        }

        # Download and read bands into memory
        bands = {}
        
        if not os.path.exists(f"./data/{self.insee}/{year}"):
            os.makedirs(f"./data/{self.insee}/{year}")

        if not os.path.exists(f'./data/{self.insee}/{year}/RGBNIR.tif'):
            for band, url in band_urls.items():
                print(f"Downloading {band} band ...")
                response = requests.get(url, stream=True)
                with rasterio.MemoryFile(response.content) as memfile:
                    with memfile.open() as src:
                        bands[band] = src.read(1)
                        if band == "swir1":
                            swir1_transform = src.transform  # Save SWIR1 transformation res 20m
                        if band == "red":
                            high_res_transform = src.transform # Save red transformation res 10m
            # Upsample swir1 from 20m res to 10m res
            if "swir1" in bands:
                high_res_shape = (bands["nir"].shape[0], bands["nir"].shape[1])  # Taille cible (10m)
                swir1_upsampled = np.empty(high_res_shape, dtype=bands["swir1"].dtype)
                # Reprojection and upsampling
                rasterio.warp.reproject(
                    source=bands["swir1"],
                    destination=swir1_upsampled,
                    src_transform=swir1_transform,
                    src_crs=src.crs,
                    dst_transform=high_res_transform,  # Transform bande red
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )
                bands["swir1"] = swir1_upsampled

            # Stack bands into a single array
            stacked_array = np.stack([bands["red"], bands["green"], bands["blue"], bands["nir"], bands["swir1"]], axis=0)
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "count": 5,
                "height": bands["red"].shape[0],  # Hauteur correspondant à 10m
                "width": bands["red"].shape[1],   # Largeur correspondant à 10m
                "transform": high_res_transform,  # Transformation des bandes à 10m
                "dtype": stacked_array.dtype
            })
            print("Saving RGB+NIR+SWIR image ...")
            output_path = os.path.join(f"./data/{self.insee}/{year}", f"RGBNIRSWIR.tif")
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(stacked_array)
                self.EPSG = dest.crs.to_epsg()
        else:
            output_path = os.path.join(f"./data/{self.insee}/{year}", f"RGBNIRSWIR.tif")
            with rasterio.open(output_path) as src:
                self.EPSG = src.crs.to_epsg()
            print(f"RGB+NIR+SWIR image already exists at {output_path}")

        print(f"RGB+NIR image saved as {output_path}")

        #Convert bbox to format of TIFF file
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.EPSG}")
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
        self.out_image_shape = out_image.shape[1:3]
        self.out_transform = out_transform
        self.out_meta = out_meta

        output_cropped_path = os.path.join(f"./data/{self.insee}/{year}", f"RGBNIRSWIR_cropped.tif")
        with rasterio.open(output_cropped_path, "w", **out_meta) as dest:
            dest.write(out_image)

        self.save_to_numpy(out_image, f"{year}/RGBNIRSWIR_cropped")
        print(f"Cropped image saved as {output_cropped_path}")

    def rasterize_houses(self, houses, out_image_shape, out_transform, out_meta, EPSG):
        #Rasterize houses geometry
        # Convert houses GeoJSON to the correct coordinate system
        houses_32631 = []
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}")
        for feature in houses.data['features']:
            geom = feature['geometry']
            new_coords = []
            for polygon in geom['coordinates']:
                new_polygon = []
                for coord in polygon[0]:
                    x, y = transformer.transform(coord[1], coord[0])
                    new_polygon.append([x, y])
                new_coords.append([new_polygon])
            geom['coordinates'] = new_coords
            houses_32631.append((geom, 1))
            # Rasterize houses geometry
            house_mask = rasterio.features.rasterize(
                houses_32631,
                out_shape=out_image_shape,
                transform=out_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
        
        mask_path = os.path.join(f"./data/{self.insee}", f"houses_mask.tif")
        mask_meta = out_meta.copy()
        mask_meta.update(count=1, dtype=rasterio.float32, width=out_image_shape[1], height=out_image_shape[0])

        with rasterio.open(mask_path, "w", **mask_meta) as dest:
            dest.write(house_mask, 1)

        self.save_to_numpy(house_mask, "houses_mask")
        print(f"House mask saved as {mask_path}")

    def save_to_numpy(self, raster, name):
        # Save the raster to a NumPy array
        np.save(f"./data/{self.insee}/{name}.npy", raster)

if __name__ == "__main__":
    # Paths to before and after databases
    db_path_b = "./bordeaux_2018.sqlite"
    db_path_a = "./bordeaux_2024.sqlite"
    
    # Create changeFinder object and find changes
    cf = changeFinder(db_path_b, db_path_a)
    ROI_dict = cf.find_changes()

    print(f"Keeping {len(ROI_dict)} regions of interest.")
    # Show geometries of the regions of interest on a map
    #cf.generate_map(ROI_dict)

    #Get map for both years
    for y in cf.years:
        cf.get_cropped_map(ROI_dict,y)
    
    cf.rasterize_houses(cf.houses, cf.out_image_shape, cf.out_transform, cf.out_meta, cf.EPSG)
    
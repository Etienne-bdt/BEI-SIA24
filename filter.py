import sqlite3

import folium
from lambert import Lambert93, convertToWGS84Deg
from shapely.geometry import mapping
from shapely.wkt import loads

#TODO : #For each bbox create a small grid of the size of the bbox and create a building mask corresponding to the building geometry
#TODO : Link Eodag to fetch sentinel-2 data around the building automatically ?

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
    Class to find changes between before and after databases.
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
            if a.batiment_rid not in before_rids:
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
        return ROI[:int(0.2*length)]
        
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
        print(ROI[list(ROI.keys())[0]])
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
                
                # Add bounding box
                bounds = loads(geom).bounds
                bbox = [
                    [bounds[1], bounds[0]],
                    [bounds[1], bounds[2]],
                    [bounds[3], bounds[2]],
                    [bounds[3], bounds[0]],
                    [bounds[1], bounds[0]]
                ]
                print(bbox)
                bbox_wgs84 = [[convertToWGS84Deg(pt[1], pt[0], Lambert93).getY(), convertToWGS84Deg(pt[1], pt[0], Lambert93).getX()] for pt in bbox]
                folium.PolyLine(bbox_wgs84, color="yellow", weight=2.5, opacity=1).add_to(m)
                
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
                    
                    # Add bounding box
                    bounds = loads(geom).bounds
                    bbox = [
                        [bounds[1], bounds[0]],
                        [bounds[1], bounds[2]],
                        [bounds[3], bounds[2]],
                        [bounds[3], bounds[0]],
                        [bounds[1], bounds[0]]
                    ]
                    bbox_wgs84 = [[convertToWGS84Deg(pt[0], pt[1], Lambert93).getY(), convertToWGS84Deg(pt[0], pt[1], Lambert93).getX()] for pt in bbox]
                    folium.PolyLine(bbox_wgs84, color="yellow", weight=2.5, opacity=1).add_to(m)
                    
        # Save the map to an HTML file and display
        output_file = "map.html"
        m.save(output_file)
        print(f"Map saved as {output_file}. Open this file in a browser to view the geometries.")

if __name__ == "__main__":
    # Paths to before and after databases
    db_path_b = "./escalquens_2018.sqlite"
    db_path_a = "./escalquens_2024.sqlite"
    
    # Create changeFinder object and find changes
    cf = changeFinder(db_path_b, db_path_a)
    ROI_dict = cf.find_changes()
    print(f"Keeping {len(ROI_dict)} regions of interest.")
    #Show geometries of the regions of interest on a map
    #This can be done using folium and mapping
    cf.generate_map(ROI_dict)
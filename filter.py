import sqlite3
from shapely.wkt import loads

#TODO : List of POI with building mask and region of the parcelle and bounding box around the building to fetch sentinel-2 data
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
        return self.filter_on_area(ROI)

    def filter_on_area(self, ROI):
        """
        Sort ROI on area of the region using a key (we don't have to store the areas in another array as such).
        We then keep the top 20% of the regions.
        """
        ROI.sort(key=lambda x: self.get_r_area(x), reverse=True)
        length = len(ROI)
        return ROI[:int(0.2*length)]
        
    def get_r_area(self, region):
        """
        Get area of a region.
        """
        geom = loads(region.geom_parcelle)
        return geom.area

if __name__ == "__main__":
    # Paths to before and after databases
    db_path_b = "./escalquens_2018.sqlite"
    db_path_a = "./escalquens_2024.sqlite"
    
    # Create changeFinder object and find changes
    cf = changeFinder(db_path_b, db_path_a)
    ROI = cf.find_changes()
    print(f"Keeping {len(ROI)} regions of interest.")
    

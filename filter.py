import sqlite3

class temporality():
    def __init__(self):
        self.batiment_rid = None
        self.parcelle_rid = None
        self.geom_batiment = None
        self.geom_parcelle = None        

class changeFinder():
    """
    Class to find changes between before and after databases.
    """
    def __init__(self, db_path_b, db_path_a):
        self._conn_b = sqlite3.connect(db_path_b)
        self._conn_b.enable_load_extension(True)
        self._conn_b.load_extension("mod_spatialite")
        self._cursor_b = self._conn_b.cursor()

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
        query = "SELECT ST_AsText(b.geom) AS batiment_geom,ST_AsText(p.geom) AS parcelle_geom, b.object_rid as batiment_rid, p.object_rid as parcelle_rid FROM geo_batiment AS b JOIN geo_batiment_parcelle AS bp ON b.geo_batiment = bp.geo_batiment JOIN geo_parcelle AS p ON bp.geo_parcelle = p.geo_parcelle;"
        self._cursor_b.execute(query)
        rows_b = self._cursor_b.fetchall()
        self._cursor_a.execute(query)
        rows_a = self._cursor_a.fetchall()
        
        for row in rows_b:
            t = temporality()
            t.parcelle_rid = row[3]
            t.batiment_rid = row[2]
            t.geom_batiment = row[0]
            t.geom_parcelle = row[1]
            before.append(t)
        for row in rows_a:
            t = temporality()
            t.parcelle_rid = row[3]
            t.batiment_rid = row[2]
            t.geom_batiment = row[0]
            t.geom_parcelle = row[1]
            after.append(t)
        return before,after
    
    def close(self):
        self._conn_b.close()
        self._conn_a.close()
    
    def find_changes(self):
        """
        Find changes between two databases.
        """
        before, after = self.get_table()
        num_changes = 0
        before_rids = [t.batiment_rid for t in before]
        after_rids = [t.batiment_rid for t in after]
        for a in after_rids:
            if a not in before_rids:
                num_changes += 1
        self.close()
        print(f"Found {num_changes} changes between the two databases.")
    
if __name__ == "__main__":
    db_path_b = "./escalquens_2018.sqlite"
    db_path_a = "./escalquens_2024.sqlite"
    cf = changeFinder(db_path_b, db_path_a)
    cf.find_changes()

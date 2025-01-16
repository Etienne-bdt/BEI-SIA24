import os

from torch.utils.data import Dataset
import pickle

class CadastreSen2Dataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.impath = image_path
        self.transform = transform
        self.products = ["RGBNIR_cropped", "houses_mask"]

    def list_data(self):
        """Naming convention will be as follows: 
        - {INSEE_code}/{year}/{band/index}.tif
        """
        cache_file = os.path.join(self.impath, "data_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.data_list = pickle.load(f)
                return self.data_list
        
        path_to_check = [x for x in os.listdir(self.impath) if os.path.isdir(os.path.join(self.impath, x)) and x.isnumeric()]
        print(path_to_check)
        for path in path_to_check:
            years = [x for x in os.listdir(os.path.join(self.impath, path)) if os.path.isdir(os.path.join(self.impath, path, x)) and x.isnumeric()]
            if len(years) != 2:
                path_to_check.remove(path)
                print(f"Error: found {len(years)} years in {path}, expected 2.")
            else:
                # Check number of files in each year
                products1 = [x.strip('.tif') for x in os.listdir(os.path.join(self.impath, path, years[0])) if x.endswith(".tif")]
                y1len = len(products1)
                if y1len != len(self.products):
                    missing = [x for x in self.products if x not in products1]
                    print(f"Missing {missing} products in {path}/{years[0]}")
                    path_to_check.remove(path)
                    break
                products2 = [x.strip('.tif') for x in os.listdir(os.path.join(self.impath, path, years[1])) if x.endswith(".tif")]
                y2len = len(products2)
                if y2len != len(self.products):
                    missing = [x for x in self.products if x not in products2]
                    print(f"Missing {missing} products in {path}/{years[1]}")
                    path_to_check.remove(path)
                    break
                if products1 != products2:
                    print(f"Products in {path}/{years[0]} and {path}/{years[1]} do not match.")
                    path_to_check.remove(path)
                    break
        with open(cache_file, "wb") as f:
            pickle.dump(path_to_check, f)
            self.data_list = path_to_check
        return self.data_list


    def __len__(self):
        if not hasattr(self, "data_list"):
            return len(self.list_data())
        else:
            return len(self.data_list)
        
    def __getitem__(self, idx):
        if not hasattr(self, "data_list"):
            self.list_data()
        if idx >= len(self.data_list):
            raise IndexError("Index out of bounds")
        
        path = self.data_list[idx]
        



if __name__ == "__main__":
    ds = CadastreSen2Dataset("data/")
    next(iter(ds))

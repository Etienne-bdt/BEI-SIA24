import os

from torch.utils.data import Dataset
import pickle
import numpy as np 
import torchvision.transforms.functional as TVF

class CadastreSen2Dataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.impath = image_path
        self.transform = transform
        self.products = ["RGBNIR_cropped", "houses_mask"]
        
        #Initialize the data list with the available data
        self.list_data()

    def list_data(self):
        """Naming convention will be as follows: 
        - {INSEE_code}/{year}/{band/index}.tif
        """
        data_dict = {}
        cache_file = os.path.join(self.impath, "data_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.data_list = pickle.load(f)
                self.keys = list(self.data_list.keys())
                return self.data_list
        
        path_to_check = [x for x in os.listdir(self.impath) if os.path.isdir(os.path.join(self.impath, x)) and x.isnumeric()]
        for path in path_to_check:
            years = [x for x in os.listdir(os.path.join(self.impath, path)) if os.path.isdir(os.path.join(self.impath, path, x)) and x.isnumeric()]
            if len(years) != 2:
                path_to_check.remove(path)
                print(f"Error: found {len(years)} years in {path}, expected 2.")
            else:
                for y in years:
                    products = [x for x in os.listdir(os.path.join(self.impath, path, y)) if x.endswith(".npy")]
                    if len(products) == 0:
                        print(f"No numpy products found in {path}/{y}")
                        path_to_check.remove(path)
                        break
                    if data_dict.get(path) is None:
                        data_dict[path] = [f"{path}/{y}/{p}" for p in products]
                    else : 
                        data_dict[path].extend([f"{path}/{y}/{p}" for p in products])

        #Cache the data list for future use (not really necessary ??)
        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
            self.data_list = path_to_check
            self.keys = list(data_dict.keys())
        return self.data_list

    def __len__(self):
        if not hasattr(self, "data_list"):
            return len(self.list_data())
        else:
            return len(self.data_list)

    def plot(self, idx, bands=[1,2,3]):
        before, after, mask = self[idx]
        import matplotlib.pyplot as plt
        #stack the 3 bands
        be = np.zeros((before.shape[1], before.shape[2], 3))
        af = np.zeros((after.shape[1], after.shape[2], 3))
        max_before = np.max(before[bands,...])
        max_after = np.max(after[bands,...])
        min_before = np.min(before[bands,...])
        min_after = np.min(after[bands,...])
        for i, b in enumerate(bands):
            be[:,:,i] = (before[b-1]-min_before)/(max_before-min_before)
            af[:,:,i] = (after[b-1]-min_after)/(max_after-min_after)
        mask = mask[0,...]
        plt.figure()
        plt.subplot(131)
        plt.imshow(be)
        plt.colorbar()
        plt.title("Before")
        plt.subplot(132)
        plt.imshow(af)
        plt.colorbar()
        plt.title("After")
        plt.subplot(133)
        plt.imshow(mask)
        plt.title("Mask")
        plt.show()

    def random_crop(self, img1, img2, cm, size):
        x = np.random.randint(0, img1.shape[2]-size)
        y = np.random.randint(0, img1.shape[1]-size)
        img1 = img1[:,y:y+size, x:x+size]
        img2 = img2[:,y:y+size, x:x+size]
        cm = cm[0:1, y:y+size, x:x+size]
        return img1, img2, cm

    def random_flip(self, img1,img2,cm, chance=0.5):
        
        if (np.random.randint(0,1)> chance):
            img1 = TVF.hflip(img1)
            img2 = TVF.hflip(img2)
            cm = TVF.hflip(cm)

        if (np.random.randint(0,1)> chance):
            img1 = TVF.vflip(img1)
            img2 = TVF.vflip(img2)
            cm = TVF.vflip(cm)

        return img1, img2, cm

    def __getitem__(self, idx):
        if not hasattr(self, "data_list"):
            self.list_data()
        if idx >= len(self.data_list):
            raise IndexError("Index out of bounds")
        
        key = self.keys[idx]

        paths = self.data_list.get(key)

        insee_code = paths[0].split("/")[0]

        before  = np.load(os.path.join(self.impath, paths[0]))
        after = np.load(os.path.join(self.impath, paths[1]))
        house_mask = np.load(os.path.join(self.impath, f"{insee_code}/houses_mask.npy"))
        house_mask = house_mask[np.newaxis,...]
        print(f"Before: {before.shape}, After: {after.shape}, Mask: {house_mask.shape}")

        if self.transform:
            before, after = self.transform(before, after)

        #Cropping and flipping randomly
        before, after, house_mask = self.random_crop(before, after, house_mask, 64)
        before, after, house_mask = self.random_flip(before, after, house_mask)
        
        #TODO: Normalize the images using quantiles
        
        return before, after, house_mask


if __name__ == "__main__":
    ds = CadastreSen2Dataset("data/")
    ds.plot(0)


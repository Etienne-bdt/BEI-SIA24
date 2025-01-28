import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset
from utils.index_calculation import NDVI, NDWI, NDBI, NDMI, BSI

class CadastreSen2Dataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.impath = image_path
        self.transform = transform
        #Initialize the data list with the available data
        #self.list_data()
        self.load_patches()

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
        mask = mask[0,...]
        plt.figure()
        plt.subplot(131)
        plt.imshow(before[bands-np.ones_like(bands),...].transpose(1,2,0))
        plt.colorbar()
        plt.title("Before")
        plt.subplot(132)
        plt.imshow(after[bands-np.ones_like(bands),...].transpose(1,2,0))
        plt.colorbar()
        plt.title("After")
        plt.subplot(133)
        plt.imshow(mask,cmap="gray")
        plt.title("Mask")
        plt.show()

        plt.figure()
        plt.subplot(131)
        plt.imshow(before[5,...], cmap="gray")
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(before[6,...], cmap="gray")
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(before[7,...], cmap="gray")
        plt.colorbar()
        plt.show()

    def create_patches(self, patch_size=64):
        if not hasattr(self, "data_list"):
            self.list_data()
        for key in self.keys:
            paths = self.data_list.get(key)
            before  = np.load(os.path.join(self.impath, paths[0]))
            after = np.load(os.path.join(self.impath, paths[1]))
            (before, after) =  (self.normalize(before), self.normalize(after))
            print(f"Before: {before.shape}, After: {after.shape}")
            print(f"Creating patches for {key}")
            NDVI_before, NDWI_before, NDBI_before, NDMI_before, BSI_before = NDVI(before), NDWI(before), NDBI(before), NDMI(before), BSI(before)
            NDVI_after, NDWI_after, NDBI_after, NDMI_after, BSI_after = NDVI(after), NDWI(after), NDBI(after), NDMI(after), BSI(after)
            before = np.concatenate([before, NDVI_before[np.newaxis,...], NDWI_before[np.newaxis,...], NDBI_before[np.newaxis,...], NDMI_before[np.newaxis,...], BSI_before[np.newaxis,...]], axis=0)
            after = np.concatenate([after, NDVI_after[np.newaxis,...], NDWI_after[np.newaxis,...], NDBI_after[np.newaxis,...], NDMI_after[np.newaxis,...], BSI_after[np.newaxis,...]], axis=0)
            print(f"Before: {before.shape}, After: {after.shape}")
            house_mask = np.load(os.path.join(self.impath, f"{key}/houses_mask.npy"))
            house_mask = house_mask[np.newaxis,...]
            os.makedirs(os.path.join(self.impath, f"{key}/patches"), exist_ok=True)
            for i in range(0, before.shape[1], patch_size):
                for j in range(0, before.shape[2], patch_size):
                    mask_patch = house_mask[:,i:i+patch_size,j:j+patch_size]
                    if mask_patch.sum() > 64*64*0.05:
                        before_patch = before[:,i:i+patch_size,j:j+patch_size]
                        after_patch = after[:,i:i+patch_size,j:j+patch_size] 
                        np.save(os.path.join(self.impath, f"{key}/patches/{i}_{j}_before.npy"), before_patch)
                        np.save(os.path.join(self.impath, f"{key}/patches/{i}_{j}_after.npy"), after_patch)
                        np.save(os.path.join(self.impath, f"{key}/patches/{i}_{j}_mask.npy"), mask_patch)

    def random_crop(self, img1, img2, cm, size):
        x = np.random.randint(0, img1.shape[2]-size)
        y = np.random.randint(0, img1.shape[1]-size)
        cm = cm[0:1, y:y+size, x:x+size]
        img1 = img1[:,y:y+size, x:x+size]
        img2 = img2[:,y:y+size, x:x+size]
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

    def normalize(self, img):
        quantiles = np.percentile(img, [2, 98])
        img = np.clip(img, quantiles[0], quantiles[1])
        img = (img - quantiles[0])/(quantiles[1] - quantiles[0])
        return img
    
    def __getitem__(self, idx):
        if not hasattr(self, "data_list"):
            self.list_data()
        if idx >= len(self.data_list):
            raise IndexError("Index out of bounds")
        
        pb, pa, pm = self.data_list[idx]
        before  = np.load(pb)
        after = np.load(pa)
        house_mask = np.load(pm)
        print(f"Before: {before.shape}, After: {after.shape}, Mask: {house_mask.shape}")

        if self.transform:
            before, after = self.transform(before, after)

        #Cropping and flipping randomly
        #before, after, house_mask = self.random_crop(before, after, house_mask, 64)
        before, after, house_mask = self.random_flip(before, after, house_mask)
        
        #TODO: Normalize the images using quantiles
        
        return before, after, house_mask

    def load_patches(self):
        data_list = []
        path_to_check = [x for x in os.listdir(self.impath) if os.path.isdir(os.path.join(self.impath, x)) and x.isnumeric()]
        for path in path_to_check:
            patch = os.path.join(self.impath, path, "patches")
            if os.path.exists(patch):
                patches = [x for x in os.listdir(patch) if x.endswith(".npy")]
                if len(patches) == 0:
                    print(f"No numpy patches found in {patch}")
                else:
                    #We have several patches with _before _after _mask, we need to group those with the same name {i}_{j} as one key in the dict or a list
                    for p in range(0,len(patches),3):
                        data_list.append((os.path.join(patch, patches[p+1]), os.path.join(patch, patches[p]), os.path.join(patch, patches[p+2])))
            else:
                print(f"Path {path}/{patches} does not exist")
        self.data_list = data_list


if __name__ == "__main__":
    ds = CadastreSen2Dataset("data/")
    #ds.create_patches(64)
    #ds.load_patches()
    print(len(ds))
    ds.plot(2)
    

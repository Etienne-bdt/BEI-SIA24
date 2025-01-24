import numpy as np
import os



def main(data_path):
    """
    Main function to calculate the index of the data

    """

    all_insee = [x for x in os.listdir(data_path) if x.isdigit()]
    for insee in all_insee:
        all_years = [x for x in os.listdir(os.path.join(data_path, insee)) if x.isdigit()]
        for year in all_years:
            all_npy = [x for x in os.listdir(os.path.join(data_path, insee, year)) if x.endswith(".npy")]
            for npy in all_npy:
                data = np.load(os.path.join(data_path, insee, year, npy))
                

def NDVI(data):
    return (data[3, :, :] - data[0, :, :]) / (data[3, :, :] + data[0, :, :])
def NDWI(data):
    return (data[1,:,:] - data[3,:,:]) / (data[1,:,:] + data[3,:,:])
def NDBI(data):
    return (data[4,:,:] - data[3,:,:]) / (data[4,:,:] + data[3,:,:])
def NDMI(data):
    return (data[1,:,:] - data[3,:,:]) / (data[1,:,:] + data[3,:,:])
def BSI(data):
    return (data[3,:,:] - data[2,:,:]) / (data[3,:,:] + data[2,:,:])


if __name__ == '__main__':
    data_path = "./data"
    main(data_path)
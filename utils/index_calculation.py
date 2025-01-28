"""Index calculation utilitary functions."""

def NDVI(data):
    return (data[3, :, :] - data[0, :, :]) / (data[3, :, :] + data[0, :, :]+1e-5)
def NDWI(data):
    return (data[1,:,:] - data[3,:,:]) / (data[1,:,:] + data[3,:,:]+1e-5)
def NDBI(data):
    return (data[4,:,:] - data[3,:,:]) / (data[4,:,:] + data[3,:,:]+1e-5)
def NDMI(data):
    return (data[1,:,:] - data[3,:,:]) / (data[1,:,:] + data[3,:,:]+1e-5)
def BSI(data):
    return (data[3,:,:] - data[2,:,:]) / (data[3,:,:] + data[2,:,:]+1e-5)

"""Index calculation utilitary functions."""

def NDVI(data):
    if data.ndim == 4:
        return (data[:, 3:4, :, :] - data[:, 0:1, :, :]) / (data[:, 3:4, :, :] + data[:, 0:1, :, :]+1e-5)
    return (data[3, :, :] - data[0, :, :]) / (data[3, :, :] + data[0, :, :]+1e-5)

def NDWI(data):
    if data.ndim == 4:
        return (data[:, 1:2, :, :] - data[:, 3:4, :, :]) / (data[:, 1:2, :, :] + data[:, 3:4, :, :]+1e-5)
    return (data[1,:,:] - data[3,:,:]) / (data[1,:,:] + data[3,:,:]+1e-5)

def NDBI(data):
    if data.ndim == 4:
        return (data[:, 4:5, :, :] - data[:, 3:4, :, :]) / (data[:, 4:5, :, :] + data[:, 3:4, :, :]+1e-5)
    return (data[4,:,:] - data[3,:,:]) / (data[4,:,:] + data[3,:,:]+1e-5)

def NDMI(data):
    if data.ndim == 4:
        return (data[:, 1:2, :, :] - data[:, 3:4, :, :]) / (data[:, 1:2, :, :] + data[:, 3:4, :, :]+1e-5)
    return (data[1,:,:] - data[3,:,:]) / (data[1,:,:] + data[3,:,:]+1e-5)

def BSI(data):
    if data.ndim == 4:
        return (data[:, 3:4, :, :] - data[:, 2:3, :, :]) / (data[:, 3:4, :, :] + data[:, 2:3, :, :]+1e-5)
    return (data[3,:,:] - data[2,:,:]) / (data[3,:,:] + data[2,:,:]+1e-5)

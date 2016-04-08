import os

#get list of images in folder
def get_imlist(path, full=True):
    return [os.path.join(path,f) if full else f for f in os.listdir(path) if f.split('.')[-1] in ['jpg', 'png', 'gif']]


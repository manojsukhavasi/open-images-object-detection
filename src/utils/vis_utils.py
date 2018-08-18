from src.imports import *

def open_image(fn):
    """ Opens an image using OpenCV given the file path.
    Arguments:
        fn: the file path of the image
    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            if str(fn).startswith("http"):
                req = urllib.request.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32)/255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None:
                raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
    
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)

def od_image(img, xy=[0,0,0,0], txt=None):
    """Shows an image with the object boxes on it
    Arguments:
        img: numpy array of the image
        xy: list of x,y coordinate rations in order xmin, ymin, xmax, ymax
        txt: Name of the class
    Returns:
        axes
    """
    ax = show_img(img)
    h,w,_ = img.shape
    bb = [xy[0]*w, xy[1]*h, (xy[2]-xy[0])*w, (xy[3]-xy[1])*h]
    draw_rect(ax, bb)
    draw_text(ax,bb[:2], txt)
    return ax

def multi_od_image(img, ann, id2class, class2name):
    """Shows an image with all the object boxes on it
    Arguments:
        img: numpy array of the image
        ann: List of all the bounding boxes with following pattern
            [classlabel, xmin, ymin, xmax, ymax]
        ids2class: dict of ids to class ids
        class2name: dict of class ids to names
    Returns:
        axis
    """
    ax = show_img(img)
    h,w,_ = img.shape
    for obj in ann:
        c = int(obj[0])
        b = obj[1:]
        bb = [b[0]*w, b[1]*h, (b[2]-b[0])*w, (b[3]-b[1])*h]
        draw_rect(ax, bb)
        draw_text(ax,bb[:2], class2name[id2class[c]])
    
    return ax
    

def save_obj(obj, fname):
    """ Saves a python object in pickle
    Arguments:
        obj: pthon object to be saved in pickle format
        name: name of the file to be saved in data/tmp folder
    Returns: None
    """
    with open(fname, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(fname):
    """ Loads a pickle object
    Arguments:
        fname: Full path of the file
    Returns:
        python object
    """
    with open(fname, 'rb') as f:
        return pkl.load(f)

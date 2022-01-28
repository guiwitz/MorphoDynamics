import numpy as np
from scipy.ndimage import binary_fill_holes
from ..splineutils import spline_to_param_image, fit_spline
import skimage.filters
import skimage.util

def generate_circle(position=[50,50], radius=10):
    """Generate coordinates of a circle with given radius and center"""
   
    position = np.array([50,50])
    circle = position+ np.array([[radius*np.cos(x), radius*np.sin(x)] for x in np.arange(0,2*np.pi, 0.01)])

    return circle

def generate_gradient_stack(height=100, width=100, steps=40):
    """Generate a stack of images with a gradient"""

    grad_image = (np.ones((height, width))*np.arange(0,width)).T
    grad_image = grad_image/width
    vert_stack = grad_image*np.ones((steps, height, width))
    vert_stack = np.rollaxis(vert_stack,0,3)

    return vert_stack

def generate_1d_wave(steps=40, shift_steps=0):
    """Generate a 1d wave of 1 period with 0 shifted by shift_steps"""

    step_size = 2*np.pi/steps
    wave = np.sin(-(shift_steps)*step_size+2*np.pi*np.arange(0, steps)/steps)
    wave[wave<0]=0

    return wave

def modulated_gradient(width, height, steps, step_shift):
    """Generate a stack of images with a gradient modulated by a wave"""

    vert_stack = generate_gradient_stack(height=height, width=width)
    wave1 = generate_1d_wave(steps=steps, shift_steps=step_shift)
    vert_stack = vert_stack * wave1
    vert_stack = np.rollaxis(vert_stack,2,0)

    return vert_stack

def moving_edge(
    height, width, steps, step_reverse, displacement, radius,
    position=None, fixed_dist=10,
    coord_noise=0.5, disp_fact=0.1):
    """
    Generate a stack of images with a moving edge

    Parameters
    ----------
    height : int
        image height
    width : int
        image width
    steps : int
        number of frames
    step_reverse : int
        frame where edge reverses direction
    displacement : float
        amount of displacement per step
    radius : float
        radius of initial circle
    position : list, optional
        x,y position of cell, by default None
    fixed_dist : int, optional
        all points closer than this limit to the
        lowest edge position won't move, by default 10
    coord_noise : float, optional
        movement noise, by default 0.5
    disp_fact : float, optional
        displacement multiplicative factor, by default 0.1

    Returns
    -------
    image_stack: numpy.ndarray
        image stack
    """    

    if position is None:
        position =[int(height/2), int(width/2)]

    circle = generate_circle(position=position, radius=radius)
    # create a binary 1d array specifying which circle coord reamain fixed
    dist_from_point = np.array([np.linalg.norm(x-circle[0,:]) for x in circle])
    dist_from_point[dist_from_point < fixed_dist] = 0

    image_stack = np.zeros((steps, height, width))
    for i in range(steps):
    
        if i<step_reverse:
            fact = -displacement
        else:
            fact = displacement

        # generate noise for the circle coordinates
        move_noise = np.random.normal(loc=0,scale=coord_noise, size=circle.shape)
        # add directional movement for a single coordinate
        # and only apply to coordinates that are not fixed
        move_noise[:,0] += fact*dist_from_point
        # accumulate the displacement
        circle = circle + disp_fact*move_noise
        
        # compute spline and create image
        circle_s, _ = fit_spline(circle, 100)
    
        rasterized = spline_to_param_image(1000, (height,width), circle_s, deltat=0)
        image = binary_fill_holes(rasterized > -1).astype(np.uint8)
        image_stack[i,:,:] = image
    
    return image_stack

def microscopify(image, sigma_smooth=1, var_noise=0.01):
    """Add noise to an image and smooth it"""

    im_stack_gauss = skimage.filters.gaussian(image, sigma=sigma_smooth, preserve_range=True)
    im_stack_noise = skimage.util.random_noise(im_stack_gauss,'gaussian', var=var_noise)
    im_stack_noise = skimage.util.img_as_ubyte(im_stack_noise)
    return im_stack_noise

def generate_dataset(
    height, width, steps, step_reverse, displacement, radius,
    shifts, position=None, fixed_dist=10,
    coord_noise=0.5, disp_fact=0.1, var_noise=0.01, sigma_smooth=1):
    """
    Generate a complete dataset.

    Parameters
    ----------
    height : int
        image height
    width : int
        image width
    steps : int
        number of frames
    step_reverse : int
        frame where edge reverses direction
    displacement : float
        amount of displacement per step
    radius : float
        radius of initial circle
    shifts: list
        list of shifts to apply to each channel
    position : list, optional
        x,y position of cell, by default None
    fixed_dist : int, optional
        all points closer than this limit to the
        lowest edge position won't move, by default 10
    coord_noise : float, optional
        movement noise, by default 0.5
    disp_fact : float, optional
        displacement multiplicative factor, by default 0.1
    var_noise : float, optional
        noise variance, by default 0.01
    sigma_smooth : float, optional
        gaussian smoothing sigma, by default 1

    Returns
    -------
    image_masked_noisy: numpy.ndarray
        image stack
    signals_masked_noisy: list of numpy.ndarray
        each element is an image stack of the shifted signals
    """  

    im_stack = moving_edge(
        height=height, width=width, steps=steps, 
        step_reverse=step_reverse, displacement=displacement, 
        radius=radius, position=position, fixed_dist=fixed_dist,
        coord_noise=coord_noise, disp_fact=disp_fact)

    signals = [
        modulated_gradient(height=height, width=width, steps=steps, step_shift=s)
        for s in shifts]

    signals_masked_noisy = [
        microscopify(s*im_stack, var_noise=var_noise, sigma_smooth=sigma_smooth)
        for s in signals]

    image_masked_noisy = microscopify(im_stack, var_noise=var_noise, sigma_smooth=sigma_smooth)
    
    return image_masked_noisy, signals_masked_noisy

    




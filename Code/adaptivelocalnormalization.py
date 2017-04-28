import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import scipy.ndimage.filters
import skimage.exposure


class AdaptiveLocalNormalization(cellprofiler.module.ImageProcessing):
    """
    Things to keep in mind: There is a fast version and a slow version of normalization. The slow version is true to
    the algorithm put forth in the paper. However, using disc structuring elements with scipy generic filtering is
    time intensive. An alternative is to use scipy uniform filtering. It is much faster, but it only works in
    rectangular windows. The trade-off for speed can be useful, so both slow and fast have been included.

    The normalization produces both positive and negative float values. It is preferred to use uint16 ndarrays,
    so after calculation of the image it is scaled to fit within the uint16 format. This might introduce some
    unwanted variation if stitching images or processing a group of images, because each is scaled independently.
    However, if the distribution of intensities across images is similar then this shouldn't too costly. To globally
    scale a set of images would require more coding overhead...

    The radius search could be improved by adding a persistence factor. For example, choose the radius that meets the
    cutoff criteria *n* consecutive times.
    """
    module_name = "AdaptiveLocalNormalization"

    variable_revision_number = 3

    def upgrade_settings(self, setting_values, variable_revision_number,  module_name, from_matlab):
        if variable_revision_number == 1:
            setting_values += ["5", "5"]

            variable_revision_number = 2

        if variable_revision_number == 2:
            setting_values = setting_values[:4] + setting_values[6:]

            variable_revision_number = 3

        return setting_values, variable_revision_number, False

    def create_settings(self):
        super(AdaptiveLocalNormalization, self).create_settings()

        self.threshold_std_float = cellprofiler.setting.Float(
            "Standard Deviation Threshold",
            0.5,
            doc="""
            Enter a number within the range [0,1.0). This specifies a fraction of the standard deviation of the image.
            """
        )

        self.radius_min = cellprofiler.setting.Integer(
            "Minimum Radius",
            5,
            doc="""
            Enter an integer greater than 0 and less than Maximum Radius. This specifies the smallest radius allowed 
            during the radius search for each pixel.
            """
        )

        self.radius_max = cellprofiler.setting.Integer(
            "Maximum Radius",
            25,
            doc="""
            Enter an integer greater than the Minimum Radius. This specifies the largest radius allowed during the 
            radius search for each pixel.
            """
        )

        self.radius_step = cellprofiler.setting.Integer(
            "Radius Step Size",
            5,
            doc="""
            Enter an integer greater than 0. This specifies the largest radius allowed during the radius search for
            each pixel. The larger the radius the longer the run time.
            """
        )

    def settings(self):
        __settings__ = super(AdaptiveLocalNormalization, self).settings()

        return __settings__ + [
            self.threshold_std_float,
            self.radius_max,
            self.radius_min,
            self.radius_step
        ]

    def visible_settings(self):
        __settings__ = super(AdaptiveLocalNormalization, self).settings()

        return __settings__ + [
            self.threshold_std_float,
            self.radius_min,
            self.radius_max,
            self.radius_step,
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = skimage.img_as_float(x.pixel_data)

        y_data = self.adaptive_local_normalization(
            x_data,
            x.spacing,
            self.threshold_std_float.value,
            self.radius_min.value,
            self.radius_max.value,
            self.radius_step.value
        )

        y_data = skimage.img_as_float(y_data)

        y = cellprofiler.image.Image(
            dimensions=dimensions,
            image=y_data,
            parent_image=x
        )

        y_name = self.y_name.value

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def mean_filter(self, img, size):
        mean = scipy.ndimage.uniform_filter(img, size, mode="reflect")
    
        mean[mean == numpy.nan] = 0
    
        return mean
    
    def sd_filter(self, img, size):
        x1 = scipy.ndimage.uniform_filter(img, size, mode="reflect")
    
        x2 = scipy.ndimage.uniform_filter(img ** 2, size, mode="reflect")
    
        sd = numpy.sqrt(x2 - (x1 ** 2))
    
        sd[sd == numpy.nan] = 0
    
        return sd

    def normalize_image(self, img, mean_img, sd_image):
        normalized = numpy.divide(img - sd_image, mean_img)
    
        return skimage.exposure.rescale_intensity(normalized, out_range=(0.0, 1.0))
    
    def adaptive_local_normalization(self, img, spacing=(1.0, 1.0, 1.0), scale=0.4, r_min=5, r_max=50, r_step=5):
        global_sd = numpy.std(img)
    
        threshold_sd = scale * global_sd
        
        print(threshold_sd)
        
        r_sd_image = numpy.zeros_like(img)
    
        r_mean_image = numpy.zeros_like(img)
    
        for r in range(r_min, r_max, r_step):
            r = numpy.divide(r * spacing[1], spacing)
    
            sd_image = self.sd_filter(img, r)
    
            mask = numpy.logical_and(sd_image >= threshold_sd, r_sd_image == 0)
    
            if not numpy.any(mask):
                continue
    
            r_sd_image[mask] = sd_image[mask]
    
            r_mean_image[mask] = self.mean_filter(img, r)[mask]
    
            if numpy.all(r_sd_image > 0):
                
                sigma_xyz = (r_min/3, r_min/3, r_min/9)
                
                print(sigma_xyz)
                
                r_sd_image_new = scipy.ndimage.filters.gaussian_filter(r_sd_image, sigma_xyz)
        
                r_mean_image_new = scipy.ndimage.filters.gaussian_filter(r_mean_image, sigma_xyz)

                print(sigma_xyz)

                return self.normalize_image(img, r_mean_image_new, r_sd_image_new)
    
        r_max = numpy.divide(r_max * spacing[1], spacing)
    
        r_mean_image[r_sd_image == 0] = self.mean_filter(img, r_max)[r_sd_image == 0]
    
        r_sd_image[r_sd_image == 0] = self.sd_filter(img, r_max)[r_sd_image == 0]
    
        r_sd_image = numpy.maximum(r_sd_image, threshold_sd)

        sigma_xyz = (r_min/3, r_min/3, r_min/9)
                
        r_sd_image_new = scipy.ndimage.filters.gaussian_filter(r_sd_image, sigma_xyz)
        
        r_mean_image_new = scipy.ndimage.filters.gaussian_filter(r_mean_image, sigma_xyz)

        print(sigma_xyz)
    
        return self.normalize_image(img, r_mean_image_new, r_sd_image_new)

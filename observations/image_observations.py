import numpy as np
import math
import gym
import pyqtgraph as pg
from gym import spaces
from PyQt5.QtCore import QRectF
import matplotlib.pyplot as plt

### This module computes an observation based on the current camera image acquired by the robot/agent.
class ImageObservationBaseline():
    '''
    # The constructor.
    #
    # world:        the world module
    # graphicsWindow:    the main window for visualization
    # visualOutput: does the module provide visual output?
    '''
    def __init__(self, world, graphicsWindow, visualOutput=True, use_gray_scale=False):

        # store the world module reference
        self.worldModule=world
        self.graphicsWindow = graphicsWindow
        self.topologyModule=None
        self.visualOutput=visualOutput
        self.observation=None
        self.use_gray_scale = use_gray_scale
        # generate a visual display of the observation
        if self.visualOutput:
            self.layout = self.graphicsWindow.centralWidget
            self.observation_plot_viewbox = pg.ViewBox(parent=self.layout, enableMouse=False, enableMenu=False)
            self.cameraImage = pg.ImageItem()
            # the observation plots will be initialized on receiving the
            self.layout.addItem(self.observation_plot_viewbox,
                                colspan=2, rowspan=1, row=1, col=0)

            self.observation_plot_viewbox.setAspectLocked(lock=True)
            self.observation_plot_viewbox.addItem(self.cameraImage)

        self.imageDims=(84, 84, 3)
        if self.use_gray_scale:
            self.imageDims=(84, 84, 1)

    def update(self):
        '''
        # the robot's/agent's pose has changed, get the new observation by evaluating
        # information from the world module:
        # the observation is plainly the robot's camera image data
        '''
        observation=self.worldModule.envData['imageData']
        if self.use_gray_scale:
            observation = self.to_gray_scale_image(observation)

        # display the observation camera image
        if self.visualOutput:
            imageData=observation
            self.cameraImage.setOpts(axisOrder='row-major')
            # mirror the image
            self.cameraImage.setImage(imageData[::-1])
            self.cameraImage.setRect(QRectF(0.0,0.0,84,84))
            self.cameraImage.setLevels([0.0, 0.9])

        self.observation=observation

    def getObservationSpace(self):
        '''
        This function returns the observation space for the given observation class.
        '''
        observation_space = gym.spaces.Box(low=0.0, high=1.0,shape=(self.imageDims[0],self.imageDims[1],self.imageDims[2]))

        return observation_space

    def to_gray_scale_image(self, image_array):
        """
        converts a 3D image array to a 2D grayscale image array
        """
        assert len(image_array.shape) == 3, 'provided image does not match the expected shape'
        # convert to greyscale
        gray_scale_image = np.sum(image_array[:, :, :3], axis=2) / image_array.shape[2]
        # adjust contrast
        contrast = 1
        gray_scale_image = contrast * (gray_scale_image - 0.5) + 0.5
        # add the additional channel number of 1
        gray_scale_image = np.reshape(gray_scale_image, (gray_scale_image.shape[0], gray_scale_image.shape[1], 1))

        return gray_scale_image

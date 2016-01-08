#Based off of http://wiki.wxpython.org/GLCanvas
#Lots of help from http://wiki.wxpython.org/Getting%20Started
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import wx
from wx import glcanvas

from sys import exit, argv
import numpy as np
import scipy.io as sio
import scipy.spatial as spatial
from pylab import cm
import os
import math
import time
import pygame

DEFAULT_SIZE = wx.Size(1200, 800)
DEFAULT_POS = wx.Point(10, 10)
MAXPOINTS = -1

def saveImageGL(mvcanvas, filename):
	view = glGetIntegerv(GL_VIEWPORT)
	img = wx.EmptyImage(view[2], view[3] )
	pixels = glReadPixels(0, 0, view[2], view[3], GL_RGB,
		             GL_UNSIGNED_BYTE)
	img.SetData( pixels )
	img = img.Mirror(False)
	img.SaveFile(filename, wx.BITMAP_TYPE_PNG)


class LoopDittyCanvas(glcanvas.GLCanvas):
	def __init__(self, parent):
		attribs = (glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER, glcanvas.WX_GL_DEPTH_SIZE, 24)
		glcanvas.GLCanvas.__init__(self, parent, -1, attribList = attribs)	
		self.context = glcanvas.GLContext(self)
		
		self.parent = parent
		#Camera state variables
		self.size = self.GetClientSize()
		self.Fs = 22050
		self.timeOffset = 0
		
		#Main state variables
		self.MousePos = [0, 0]
		self.initiallyResized = False
		
		#Point cloud and playing information
		self.filename = ""
		self.X = np.array([])
		self.SampleDelays = np.array([])
		self.Playing = False
		self.PlayIDX = 0
		self.DrawEdges = True
		
		self.GLinitialized = False
		#GL-related events
		wx.EVT_ERASE_BACKGROUND(self, self.processEraseBackgroundEvent)
		wx.EVT_SIZE(self, self.processSizeEvent)
		wx.EVT_PAINT(self, self.processPaintEvent)
		#Mouse Events
		wx.EVT_LEFT_DOWN(self, self.MouseDown)
		wx.EVT_LEFT_UP(self, self.MouseUp)
		wx.EVT_RIGHT_DOWN(self, self.MouseDown)
		wx.EVT_RIGHT_UP(self, self.MouseUp)
		wx.EVT_MIDDLE_DOWN(self, self.MouseDown)
		wx.EVT_MIDDLE_UP(self, self.MouseUp)
		wx.EVT_MOTION(self, self.MouseMotion)		
		#self.initGL()
	
	
	def processEraseBackgroundEvent(self, event): pass #avoid flashing on MSW.

	def setup2DProjectionMatrix(self, xmin, xmax):
		#Set up projection matrix
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		aspect = float(self.size.width)/float(self.size.height)
		if self.size.width >= self.size.height:
			#gluOrtho2D(-1.0*aspect, 1.0*aspect, -1, 1)
			gluOrtho2D(xmin*aspect, xmax*aspect, -1, 1)
		else:
			gluOrtho2D(xmin, xmax, -1.0/aspect, 1.0/aspect)

	def processSizeEvent(self, event):
		self.size = self.GetClientSize()
		self.SetCurrent(self.context)
		glViewport(0, 0, self.size.width, self.size.height)
		self.setup2DProjectionMatrix(-1, 1)


	def processPaintEvent(self, event):
		dc = wx.PaintDC(self)
		self.SetCurrent(self.context)
		if not self.GLinitialized:
			self.initGL()
			self.GLinitialized = True
		self.repaint()

	def startAnimation(self, evt):
		if len(self.SampleDelays) > 0:
			print "Playing %s"%self.filename
			self.Playing = True
			self.PlayIDX = 0
			pygame.mixer.quit()
			print "Starting mixer at %i"%self.Fs
			pygame.mixer.init(frequency = self.Fs)
			pygame.mixer.music.load(self.filename)
			pygame.mixer.music.play(0, 0)
			self.Refresh()
	
	#The user can change the position in the song
	def SliderMove(self, evt):
		pos = evt.GetPosition()
		time = np.max(self.SampleDelays)*float(pos)/1000.0
		pygame.mixer.music.play(0, time)
		self.timeOffset = time
		self.PlayIDX = 0
		self.Playing = True

	def repaint(self):
		#Set up modelview matrix
		glClearColor(1.0, 1.0, 1.0, 0.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		if len(self.X) > 0:
			glDisable(GL_LIGHTING)
			glColor3f(0, 0, 0)
			glPointSize(3)
			glLineWidth(1)
			NPoints = self.X.shape[0]
			StartPoint = 0
			EndPoint = self.X.shape[0]-1
			if self.Playing:
				dT = self.timeOffset + float(pygame.mixer.music.get_pos()) / 1000.0
				sliderPos = int(np.round(1000*dT/np.max(self.SampleDelays)))
				self.timeSlider.SetValue(sliderPos)
				self.TimeTxt.SetValue("%g"%dT)
				while dT > self.SampleDelays[self.PlayIDX]:
					self.PlayIDX = self.PlayIDX + 1
					if self.PlayIDX == NPoints - 1:
						self.Playing = False
				EndPoint = self.PlayIDX+1
				if MAXPOINTS != -1:
					StartPoint = max(EndPoint - MAXPOINTS, 0)
				self.Refresh()
			if self.PlayIDX >= len(self.X):
				self.PlayIDX = len(self.X) - 1
			
			self.setup2DProjectionMatrix(-0.1, 0.1)
			
			self.vbo.bind()
			glEnableClientState(GL_VERTEX_ARRAY)
			glVertexPointerf( self.vbo )
			
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
			glTranslatef(-self.X[self.PlayIDX, 0], 0, 0)
			EndPoint = self.X.shape[0] - 1
			if self.DrawEdges:
				glDrawArrays(GL_LINES, 0, EndPoint+1)
				glDrawArrays(GL_LINES, 1, EndPoint)
			glDrawArrays(GL_POINTS, 0, EndPoint + 1)
			
			#Now draw the current point in time
			glPointSize(10)
			glLineWidth(6)
			glColor3f(1, 0, 0)
			glBegin(GL_POINTS)
			glVertex2f(self.X[self.PlayIDX, 0], self.X[self.PlayIDX, 1])
			glEnd()
			if self.DrawEdges:
				glBegin(GL_LINES)
				glVertex2f(self.X[self.PlayIDX, 0], -1)
				glVertex2f(self.X[self.PlayIDX, 0], self.X[self.PlayIDX, 1])
				glEnd()
			
			self.vbo.unbind()
			glDisableClientState(GL_VERTEX_ARRAY)
		self.SwapBuffers()
		self.Refresh()
	
	def initGL(self):		
		glutInit('')
		glEnable(GL_NORMALIZE)
		glEnable(GL_DEPTH_TEST)

	def handleMouseStuff(self, x, y):
		#Invert y from what the window manager says
		y = self.size.height - y
		self.MousePos = [x, y]

	def MouseDown(self, evt):
		x, y = evt.GetPosition()
		self.CaptureMouse()
		self.handleMouseStuff(x, y)
		self.Refresh()
	
	def MouseUp(self, evt):
		x, y = evt.GetPosition()
		self.handleMouseStuff(x, y)
		self.ReleaseMouse()
		self.Refresh()

	def MouseMotion(self, evt):
		x, y = evt.GetPosition()
		[lastX, lastY] = self.MousePos
		self.handleMouseStuff(x, y)
		dX = self.MousePos[0] - lastX
		dY = self.MousePos[1] - lastY
		#TODO: Enable motion here
		self.Refresh()

class LoopDittyFrame(wx.Frame):
	(ID_LOADMATFILE, ID_SAVESCREENSHOT, ID_EXIT) = (1, 2, 3)
	
	def ToggleDisplayEdges(self, evt):
		self.glcanvas.DrawEdges = self.EdgesToggleCheckbox.GetValue()
		self.glcanvas.Refresh()	
	
	def __init__(self, parent, id, title, pos=DEFAULT_POS, size=DEFAULT_SIZE, style=wx.DEFAULT_FRAME_STYLE, name = 'GLWindow'):
		style = style | wx.NO_FULL_REPAINT_ON_RESIZE
		super(LoopDittyFrame, self).__init__(parent, id, title, pos, size, style, name)
		#Initialize the menu
		self.CreateStatusBar()
		
		self.Fs = 22050
		self.lastIdx = np.array([])
		
		self.size = size
		self.pos = pos
		
		filemenu = wx.Menu()
		menuOpenMatfile = filemenu.Append(LoopDittyFrame.ID_LOADMATFILE, "&Load Mat File","Load Mat File")
		self.Bind(wx.EVT_MENU, self.OnLoadMatFile, menuOpenMatfile)
		menuSaveScreenshot = filemenu.Append(LoopDittyFrame.ID_SAVESCREENSHOT, "&Save Screenshot", "Save a screenshot of the GL Canvas")		
		self.Bind(wx.EVT_MENU, self.OnSaveScreenshot, menuSaveScreenshot)
		menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
		self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
		
		# Creating the menubar
		menuBar = wx.MenuBar()
		menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
		self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
		self.glcanvas = LoopDittyCanvas(self)
		
		self.rightPanel = wx.BoxSizer(wx.VERTICAL)
		
		#Buttons to go to a default view
		animatePanel = wx.BoxSizer(wx.VERTICAL)
		self.rightPanel.Add(wx.StaticText(self, label="Animation Options"), 0, wx.EXPAND)
		self.rightPanel.Add(animatePanel, 0, wx.EXPAND)
		playButton = wx.Button(self, -1, "Play")
		self.Bind(wx.EVT_BUTTON, self.glcanvas.startAnimation, playButton)
		animatePanel.Add(playButton, 0, wx.EXPAND)
		
		#Checkbox for edge toggle
		self.EdgesToggleCheckbox = wx.CheckBox(self, label="Display Edges")
		self.EdgesToggleCheckbox.SetValue(True)
		self.EdgesToggleCheckbox.Bind(wx.EVT_CHECKBOX, self.ToggleDisplayEdges)
		animatePanel.Add(self.EdgesToggleCheckbox)
		
		#Song Information		
		hbox1 = wx.BoxSizer(wx.HORIZONTAL)
		hbox1.Add(wx.StaticText(self, label='Time'))
		self.glcanvas.TimeTxt = wx.TextCtrl(self)
		self.glcanvas.TimeTxt.SetValue("0")
		hbox1.Add(self.glcanvas.TimeTxt, flag=wx.LEFT, border=5)
		animatePanel.Add(hbox1)				
		
		#Add the scroll bar to choose the time of the song
		glCanvasSizer = wx.BoxSizer(wx.VERTICAL)
		glCanvasSizer.Add(self.glcanvas, 2, wx.EXPAND)
		self.glcanvas.timeSlider = wx.Slider(self, -1, 0, 0, 1000)
		glCanvasSizer.Add(self.glcanvas.timeSlider, 0, wx.EXPAND)
		self.glcanvas.timeSlider.Bind(wx.EVT_COMMAND_SCROLL, self.glcanvas.SliderMove)		
		
		#Finally add the two main panels to the sizer
		self.sizer = wx.BoxSizer(wx.HORIZONTAL)
		self.sizer.Add(glCanvasSizer, 2, wx.EXPAND)
		self.sizer.Add(self.rightPanel, 0, wx.EXPAND)
		
		self.SetSizer(self.sizer)
		self.Layout()
		self.Show()

	#Load delay series from an external source
	def OnLoadMatFile(self, evt):
		dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.OPEN)
		if dlg.ShowModal() == wx.ID_OK:
			self.externalFile = True
			filename = dlg.GetFilename()
			dirname = dlg.GetDirectory()
			print "Loading %s...."%filename
			filepath = os.path.join(dirname, filename)
			data = sio.loadmat(filepath)
			self.Fs = data['Fs']
			self.glcanvas.Fs = self.Fs
			self.filename = filename
			
			self.glcanvas.SampleDelays = data['SampleDelays'].flatten()
			self.glcanvas.X = data['X']
			#Scale X so that it lies in the square [-1, 1] x [-1, 1]
			self.glcanvas.X = self.glcanvas.X - np.min(self.glcanvas.X, 0)
			self.glcanvas.X = self.glcanvas.X / np.max(self.glcanvas.X, 0)
			self.glcanvas.X = self.glcanvas.X * 2 - 1
			self.glcanvas.vbo = vbo.VBO(np.array(self.glcanvas.X, dtype = 'float32'))
			
			#The sound file needs to be in the same directory
			self.filename = str(data['soundfilename'][0])
			self.glcanvas.filename = self.filename
			
			self.glcanvas.Refresh()
		dlg.Destroy()
		return

	def OnSavePointCloud(self, evt):
		dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
		if dlg.ShowModal() == wx.ID_OK:
			filename = dlg.GetFilename()
			dirname = dlg.GetDirectory()
			filepath = os.path.join(dirname, filename)
			sio.savemat(filepath, {'X':self.DelaySeries})
		dlg.Destroy()
		return

	def OnSaveScreenshot(self, evt):
		dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
		if dlg.ShowModal() == wx.ID_OK:
			filename = dlg.GetFilename()
			dirname = dlg.GetDirectory()
			filepath = os.path.join(dirname, filename)
			saveImageGL(self.glcanvas, filepath)
		dlg.Destroy()
		return

	def OnExit(self, evt):
		self.Close(True)
		return

class LoopDitty(object):
	def __init__(self):
		app = wx.App()
		frame = LoopDittyFrame(None, -1, 'LoopDitty')
		frame.Show(True)
		app.MainLoop()
		app.Destroy()

if __name__ == '__main__':
	app = LoopDitty()
